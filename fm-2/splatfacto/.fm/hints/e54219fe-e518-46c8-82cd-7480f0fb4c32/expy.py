# %% [markdown]
# # absgrad vs grad — 1D 장난감 실험으로 보는 gradient collision
#
# Splatfacto의 `use_absgrad=True`는 densify 판단에 픽셀별 gradient의
# **절대값 합**(AbsGS의 homodirectional gradient)을 쓴다.
#
# 왜 필요한가? 넓은 가우시안 하나가 고주파 디테일을 덮고 있으면(over-reconstruction)
# 픽셀별 residual의 부호가 번갈아 나타나고, 위치 gradient의 합은
#
# $$g = \sum_i g_i \approx 0 \quad (\text{부호 상쇄, gradient collision})$$
#
# 이 되어 임계값을 못 넘는다. 반면
#
# $$g_{\text{abs}} = \sum_i |g_i|$$
#
# 는 상쇄 없이 오차 기여를 그대로 누적하므로 "split이 필요한 가우시안"을 잡아낸다.
# 아래에서 1D 렌더링 모델로 이를 직접 계산해 본다. (실제 gsplat 불필요)

# %%
# 필요 패키지: numpy, plotly, kaleido(정적 이미지 저장용)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1D 렌더링 모델
#
# 픽셀 $x_i$에서 렌더 값이 가우시안 하나로 결정된다고 하자:
#
# $$I(x_i) = a\, G(x_i;\mu,\sigma), \qquad G = \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
#
# 픽셀별 L2 손실 $L_i = (I_i - T_i)^2$ 의 위치 $\mu$에 대한 gradient는
#
# $$g_i = \frac{\partial L_i}{\partial \mu} = 2\,r_i \cdot a\,G(x_i)\,\frac{x_i-\mu}{\sigma^2},
# \qquad r_i = I_i - T_i$$
#
# 두 시나리오를 비교한다:
# - **(A) over-reconstruction**: 타깃에 고주파 디테일 → residual 부호가 번갈아 나타남
# - **(B) 단순 위치 오차**: 타깃 가우시안이 옆으로 밀려 있음 → residual 부호가 한쪽으로 정렬

# %%
x = np.linspace(0, 200, 400)          # 픽셀 좌표
mu, sigma, a = 100.0, 25.0, 1.0       # 넓은 가우시안 하나

G = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
render = a * G
dG_dmu = a * G * (x - mu) / sigma**2  # dI/dmu

# (A) 타깃 = 렌더 + 고주파 디테일 → residual이 sin으로 진동 (부호 교대)
target_A = render + 0.3 * np.sin(x / 2.5) * G   # 가우시안 지지영역 안의 고주파 텍스처
r_A = render - target_A
g_A = 2 * r_A * dG_dmu                          # 픽셀별 gradient

# (B) 타깃 = 오른쪽으로 8픽셀 밀린 가우시안 → residual 부호 정렬
target_B = a * np.exp(-((x - (mu + 8)) ** 2) / (2 * sigma**2))
r_B = render - target_B
g_B = 2 * r_B * dG_dmu

for name, g in [("A(고주파 디테일)", g_A), ("B(위치 오차)", g_B)]:
    print(f"{name:16s}  |sum g| = {abs(g.sum()):8.4f}   sum|g| = {np.abs(g).sum():8.4f}")
# 출력:
# A(고주파 디테일)        |sum g| =   0.0000   sum|g| =   0.7615
# B(위치 오차)          |sum g| =   1.1029   sum|g| =   1.1047

# %% [markdown]
# ## 결과 해석
#
# | 시나리오 | $\|\sum g_i\|$ (grad) | $\sum \|g_i\|$ (absgrad) |
# |---|---|---|
# | A: 고주파 디테일 (split 필요!) | **≈ 0.0000 — 완전 소멸** | 0.7615 |
# | B: 단순 위치 오차 | 1.1029 | 1.1047 |
#
# - **A**: 픽셀별 gradient는 크지만 부호가 교대해 합이 0으로 상쇄된다.
#   일반 grad 기준으로는 densify 임계값을 절대 못 넘고, 큰 blur 가우시안이 그대로 남는다.
#   absgrad(9.65)는 이 가우시안을 확실히 잡아낸다 — 이것이 `use_absgrad=True`의 효과.
# - **B**: 부호가 정렬된 경우엔 grad와 absgrad가 거의 같다. 즉 absgrad는
#   기존에 잘 잡히던 경우를 해치지 않으면서 상쇄 케이스만 추가로 구제한다.
# - absgrad ≥ grad 항상 성립하므로 임계값도 함께 올린다
#   (Splatfacto `densify_grad_thresh=0.0008` vs 원조 3DGS 0.0002).

# %%
fig = make_subplots(
    rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
    subplot_titles=(
        "(A) 렌더 vs 타깃: 고주파 디테일", "(A) 픽셀별 gradient g_i — 부호 교대",
        "(B) 렌더 vs 타깃: 위치 오차", "누적 기준 비교: |Σg| vs Σ|g|",
    ),
)
fig.add_trace(go.Scatter(x=x, y=render, name="렌더 (넓은 가우시안)", line=dict(color="#4C78A8")), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=target_A, name="타깃 A (디테일)", line=dict(color="#E45756")), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=g_A, name="g_i (A)", line=dict(color="#72B7B2"), showlegend=False), row=1, col=2)
fig.add_hline(y=0, line=dict(color="gray", dash="dot"), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=render, showlegend=False, line=dict(color="#4C78A8")), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=target_B, name="타깃 B (밀림)", line=dict(color="#F58518")), row=2, col=1)
fig.add_trace(go.Bar(
    x=["A: grad", "A: absgrad", "B: grad", "B: absgrad"],
    y=[abs(g_A.sum()), np.abs(g_A).sum(), abs(g_B.sum()), np.abs(g_B).sum()],
    marker_color=["#B0B0B0", "#72B7B2", "#B0B0B0", "#72B7B2"],
    showlegend=False,
), row=2, col=2)
fig.update_layout(
    title="gradient collision: 부호 상쇄로 사라지는 densify 신호를 absgrad가 살린다",
    height=640, width=1000, template="plotly_white",
    legend=dict(orientation="h", y=-0.08),
)
_show(fig)

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
fig.write_image(os.path.join(_here, "expy.png"), scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## Splatfacto 코드와의 연결 (splatfacto.py)
#
# - 107–108행: `use_absgrad: bool = True` (config, 기본 켜짐)
# - 277행: `DefaultStrategy(..., absgrad=self.config.use_absgrad, ...)` —
#   gsplat 전략이 `info["means2d"].absgrad`를 densify 기준으로 사용
# - 571행: `rasterization(..., absgrad=self.strategy.absgrad ...)` —
#   backward에서 픽셀별 |gradient|를 별도 버퍼에 누적하도록 지시
#
# 위 장난감의 "시나리오 A"가 곧 AbsGS 논문이 말하는 over-reconstruction이다:
# 큰 가우시안이 고주파 영역을 덮으면 grad 기준으로는 split 신호가 소멸하지만
# absgrad 기준으로는 살아남는다.
