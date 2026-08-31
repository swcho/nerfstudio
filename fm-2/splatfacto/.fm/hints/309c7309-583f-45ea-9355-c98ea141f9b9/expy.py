# %% [markdown]
# # Splatfacto의 coarse-to-fine 해상도 스케줄
#
# Splatfacto는 학습 초반에 이미지를 저해상도로 다운스케일해서 시작하고,
# 일정 스텝마다 해상도를 2배씩 올려 결국 원본 해상도로 학습하는
# **coarse-to-fine** 전략을 쓴다.
#
# 관련 config (`splatfacto.py` 93-98행):
# - `resolution_schedule` (기본 **3000**): 이 스텝 수마다 해상도가 2배가 된다
# - `num_downscales` (기본 **2**): 시작 해상도는 $1/2^d$ ($d$ = `num_downscales`)
#
# 다운스케일 계수 공식 (`_get_downscale_factor`, 432-439행):
#
# $$\text{factor} = 2^{\max(\text{num\_downscales} - \lfloor \text{step} / \text{resolution\_schedule} \rfloor,\ 0)}$$
#
# 평가(eval) 시에는 `self.training`이 False이므로 **항상 1** (원본 해상도)이다.

# %%
# 필요 패키지: plotly, kaleido (그래프 저장용; 없으면 앞 셀들만 실행 가능)
import os


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 공식을 순수 파이썬으로 구현
#
# nerfstudio 원본 코드(432-439행)를 그대로 옮긴 것. `training` 플래그에 따라
# 학습 중에만 다운스케일이 적용되는 것도 재현한다.

# %%
def get_downscale_factor(step, num_downscales=2, resolution_schedule=3000, training=True):
    """splatfacto.py의 _get_downscale_factor를 순수 파이썬으로 재현."""
    if training:
        return 2 ** max(num_downscales - step // resolution_schedule, 0)
    else:
        return 1


# %% [markdown]
# ## 2. 주요 스텝에서 계수 확인 (기본값 d=2, schedule=3000)
#
# - step 0~2999: $2^{\max(2-0,0)} = 4$ → 해상도 $1/4$
# - step 3000~5999: $2^{\max(2-1,0)} = 2$ → 해상도 $1/2$
# - step 6000 이후: $2^{\max(2-2,0)} = 1$ → 원본 해상도
# - `max(..., 0)` 덕분에 step이 아무리 커져도(예: 30000, 몫=10) 계수는 1 밑으로 내려가지 않는다

# %%
for step in [0, 2999, 3000, 5999, 6000, 30000]:
    f = get_downscale_factor(step)
    print(f"step {step:>5}: downscale factor = {f}  (해상도 = 1/{f})")

print("eval  (training=False):", get_downscale_factor(0, training=False))
# 출력:
# step     0: downscale factor = 4  (해상도 = 1/4)
# step  2999: downscale factor = 4  (해상도 = 1/4)
# step  3000: downscale factor = 2  (해상도 = 1/2)
# step  5999: downscale factor = 2  (해상도 = 1/2)
# step  6000: downscale factor = 1  (해상도 = 1/1)
# step 30000: downscale factor = 1  (해상도 = 1/1)
# eval  (training=False): 1

# %% [markdown]
# ## 3. 30k 스텝 동안 유효 해상도 비율 그래프
#
# 유효 해상도 비율 = $1/\text{factor}$ (한 축 기준).
# 계단(step) 형태로 $1/4 \to 1/2 \to 1$ 순서로 올라간다.
# 비교를 위해 `num_downscales=3`(시작 $1/8$)인 경우도 함께 그린다.

# %%
import plotly.graph_objects as go

steps = list(range(0, 30001, 50))

fig = go.Figure()
for d, color in [(2, "#1f77b4"), (3, "#ff7f0e")]:
    frac = [1 / get_downscale_factor(s, num_downscales=d) for s in steps]
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=frac,
            mode="lines",
            line_shape="hv",  # 계단형
            name=f"num_downscales={d}" + (" (기본)" if d == 2 else ""),
            line=dict(color=color, width=3),
        )
    )

for x in [3000, 6000, 9000]:
    fig.add_vline(x=x, line_dash="dot", line_color="gray", opacity=0.5)

fig.update_layout(
    title="Splatfacto coarse-to-fine: 유효 해상도 비율 (resolution_schedule=3000)",
    xaxis_title="학습 step",
    yaxis_title="유효 해상도 비율 (1/factor)",
    yaxis=dict(tickvals=[0.125, 0.25, 0.5, 1.0], tickformat=".3g"),
    legend=dict(x=0.55, y=0.15),
    template="plotly_white",
)

_show(fig)

png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(png_path, width=900, height=500, scale=2)
print("saved:", png_path)
# 출력:
# saved: .../.fm/hints/309c7309-583f-45ea-9355-c98ea141f9b9/expy.png

# %% [markdown]
# ## 4. 왜 coarse-to-fine이 초기 학습을 안정화하는가
#
# - **초기 가우시안은 크고 배치가 부정확하다**: SfM 포인트에서 초기화된 가우시안은
#   아직 씬의 대략적 구조만 담고 있다. 원본 해상도의 고주파 디테일(텍스처, 엣지)에
#   바로 맞추려 하면 그래디언트가 노이즈에 지배되어 가우시안이 잘못된 위치로
#   끌려가거나 과도하게 분열(densify)된다.
# - **저해상도 = 저주파 타깃**: $1/4$ 해상도 이미지는 사실상 저역 통과 필터를 거친
#   타깃이라, 초반에는 큰 가우시안 몇 개로 씬의 전체 구조(색 분포, 대략적 형상)를
#   먼저 맞추게 된다. 최적화가 부드러운 손실 지형에서 시작하는 셈이다.
# - **점진적 디테일 추가**: 3000 스텝마다 해상도가 2배가 되면서, 구조가 잡힌 뒤에야
#   고주파 성분이 손실에 들어온다. 이때부터 refinement(split/duplicate)가 세밀한
#   디테일을 담당할 작은 가우시안을 만들어낸다.
# - **부수 효과 — 속도**: 초반 6000 스텝 동안 렌더링 해상도가 낮아 픽셀 수가
#   $1/16 \sim 1/4$이므로 학습 초반이 훨씬 빠르다.
# - **평가는 항상 원본 해상도**: eval 메트릭이 스케줄에 오염되지 않도록
#   `training=False`일 때는 무조건 factor 1을 반환한다.

# %%
# 픽셀 수 절감 효과 확인: factor f 다운스케일 시 픽셀 수는 1/f^2
for step in [0, 3000, 6000]:
    f = get_downscale_factor(step)
    print(f"step {step:>4}: factor {f} -> 픽셀 수 원본 대비 1/{f*f}")
# 출력:
# step    0: factor 4 -> 픽셀 수 원본 대비 1/16
# step 3000: factor 2 -> 픽셀 수 원본 대비 1/4
# step 6000: factor 1 -> 픽셀 수 원본 대비 1/1
