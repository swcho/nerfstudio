# %% [markdown]
# # `sh_degree` / `sh_degree_interval` — SH 차수 스케줄 이해하기
#
# Splatfacto(`nerfstudio/models/splatfacto.py`)의 두 설정값:
#
# - `sh_degree: int = 3` — 사용할 spherical harmonics(SH)의 **최대 차수**
# - `sh_degree_interval: int = 1000` — SH 차수를 **한 단계씩 추가로 켜는 주기**(스텝 수)
#
# `get_outputs()` 안에서 실제 렌더링에 쓰는 차수는 다음 한 줄로 결정된다:
#
# ```python
# sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
# ```
#
# 수식으로 쓰면:
#
# $$d(t) = \min\!\left(\left\lfloor \frac{t}{\text{interval}} \right\rfloor,\; d_{\max}\right)$$
#
# 즉 0차(상수항, 뷰 방향 무관 색)부터 시작해 1000스텝마다 한 차수씩 늘어나고,
# 3000스텝 이후에는 `sh_degree=3`에 고정된다. 이는 coarse-to-fine 전략으로,
# 초반에는 기본 색(diffuse)을 먼저 안정적으로 학습하고
# 이후 점진적으로 뷰 의존적(view-dependent) 성분을 추가한다.

# %%
# 필요 패키지: plotly, kaleido (pip install plotly kaleido)
import os

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

# %% [markdown]
# ## 1. 스케줄 함수 그대로 구현해 보기
#
# splatfacto의 한 줄을 순수 파이썬 함수로 옮기면:

# %%
def sh_degree_to_use(step: int, sh_degree: int = 3, sh_degree_interval: int = 1000) -> int:
    """splatfacto get_outputs()의 계산과 동일"""
    return min(step // sh_degree_interval, sh_degree)

for step in [0, 500, 999, 1000, 1500, 2000, 2999, 3000, 5000, 30000]:
    print(f"step={step:>6} -> 사용 차수 d = {sh_degree_to_use(step)}")
# 출력:
# step=     0 -> 사용 차수 d = 0
# step=   500 -> 사용 차수 d = 0
# step=   999 -> 사용 차수 d = 0
# step=  1000 -> 사용 차수 d = 1
# step=  1500 -> 사용 차수 d = 1
# step=  2000 -> 사용 차수 d = 2
# step=  2999 -> 사용 차수 d = 2
# step=  3000 -> 사용 차수 d = 3
# step=  5000 -> 사용 차수 d = 3
# step= 30000 -> 사용 차수 d = 3

# %% [markdown]
# ## 2. 차수별로 활성화되는 SH 계수 개수
#
# 차수 $d$까지의 SH basis 개수는 $(d+1)^2$ (RGB 각 채널마다 이만큼의 계수).
#
# | 차수 $d$ | 새로 추가되는 basis $2d+1$ | 누적 basis $(d+1)^2$ |
# |---|---|---|
# | 0 | 1 | 1 |
# | 1 | 3 | 4 |
# | 2 | 5 | 9 |
# | 3 | 7 | 16 |
#
# 파라미터는 `dim_sh = num_sh_bases(sh_degree)`로 **처음부터 최대 크기(16)로 할당**되지만
# (`features_dc` 1개 + `features_rest` 15개), 렌더링 시 `sh_degree_to_use`에 따라
# 앞쪽 $(d+1)^2$개만 사용된다 — 나머지는 gradient가 흐르지 않아 사실상 꺼진 상태.

# %%
def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2

for d in range(4):
    print(f"d={d}: 누적 basis={num_sh_bases(d):>2}, RGB 계수={num_sh_bases(d)*3:>2}개/가우시안")
# 출력:
# d=0: 누적 basis= 1, RGB 계수= 3개/가우시안
# d=1: 누적 basis= 4, RGB 계수=12개/가우시안
# d=2: 누적 basis= 9, RGB 계수=27개/가우시안
# d=3: 누적 basis=16, RGB 계수=48개/가우시안

# %% [markdown]
# ## 3. 학습 스텝에 따른 스케줄 시각화
#
# 왼쪽: 사용 차수 $d(t)$ (계단형). 오른쪽: 활성 SH basis 수 $(d(t)+1)^2$ / 최대 16.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SH_DEGREE, INTERVAL, MAX_STEP = 3, 1000, 6000
steps = list(range(0, MAX_STEP + 1, 10))
degrees = [sh_degree_to_use(s, SH_DEGREE, INTERVAL) for s in steps]
bases = [num_sh_bases(d) for d in degrees]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("사용 SH 차수 d(t)", "활성 SH basis 수 (d+1)²"),
)
fig.add_trace(go.Scatter(x=steps, y=degrees, mode="lines", name="d(t)",
                         line=dict(shape="hv", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=steps, y=bases, mode="lines", name="(d+1)²",
                         line=dict(shape="hv", width=3, color="crimson")), row=1, col=2)
fig.add_hline(y=SH_DEGREE, line_dash="dot", line_color="gray", row=1, col=1,
              annotation_text="sh_degree=3 상한")
fig.add_hline(y=16, line_dash="dot", line_color="gray", row=1, col=2,
              annotation_text="최대 16 basis")
for k in range(1, SH_DEGREE + 1):
    fig.add_vline(x=k * INTERVAL, line_dash="dash", line_color="lightgray", row=1, col=1)
    fig.add_vline(x=k * INTERVAL, line_dash="dash", line_color="lightgray", row=1, col=2)
fig.update_layout(
    title="sh_degree=3, sh_degree_interval=1000 스케줄: min(step // 1000, 3)",
    width=950, height=420, showlegend=False,
)
fig.update_xaxes(title_text="학습 step", row=1, col=1)
fig.update_xaxes(title_text="학습 step", row=1, col=2)
fig.update_yaxes(title_text="차수 d", row=1, col=1)
fig.update_yaxes(title_text="basis 수", row=1, col=2)

_show(fig)
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
fig.write_image(os.path.join(_here, "expy.png"), scale=2)
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료

# %% [markdown]
# ## 4. 요약
#
# - `sh_degree`(기본 3): SH 최대 차수 → 가우시안당 색 계수 $(d_{\max}+1)^2 \times 3 = 48$개 할당
# - `sh_degree_interval`(기본 1000): 한 차수를 추가로 켜기까지의 스텝 수
# - 실제 사용 차수: $d(t)=\min(\lfloor t/1000 \rfloor, 3)$ — 0차부터 시작, 1000스텝마다 +1, 3000스텝에 최대 도달
# - 효과: 초반엔 diffuse 색부터 안정적으로 맞추고, 뷰 의존 성분(고차 SH)은 점진적으로 학습
# - `sh_degree=0`이면 SH를 아예 쓰지 않고 `sigmoid(features_dc)`로 색만 최적화
