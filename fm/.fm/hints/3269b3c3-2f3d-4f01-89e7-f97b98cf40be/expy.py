# %% [markdown]
# # absgrad vs grad — 1D 토이로 보는 densification 신호의 상쇄
#
# 가우시안 하나의 화면 위치 $\mu$ 는 여러 픽셀 $p$ 에 기여합니다. autograd가 남기는 일반 grad는 픽셀별 기여의 **부호 있는 합**이고,
# gsplat `absgrad=True` 가 남기는 absgrad는 픽셀별 기여의 **절댓값 합**입니다:
#
# $$
# \text{grad} = \sum_p \frac{\partial \mathcal L}{\partial \mu}\Big|_p ,\qquad
# \text{absgrad} = \sum_p \left|\frac{\partial \mathcal L}{\partial \mu}\Big|_p\right| \;\ge\; |\text{grad}|
# $$
#
# 여기서는 "큰 가우시안 하나가 떨어진 GT 봉우리 두 개를 덮는" 상황을 numpy 로 만들고, 두 지표가 어떻게 갈리는지 봅니다.
# 필요 패키지: numpy, plotly, kaleido

# %%
import os
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


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. 장면 구성 — 픽셀 격자, GT(두 봉우리), 모델(넓은 가우시안 하나)
#
# 픽셀 $x_p$ 위에 모델 $f(x_p;\mu) = a\exp\!\big(-\tfrac{(x_p-\mu)^2}{2s^2}\big)$, GT 는 $\pm 3$ 에 있는 좁은 봉우리 둘. 손실은 픽셀별 제곱오차 $\mathcal L = \sum_p \ell_p,\ \ell_p = (f_p - g_p)^2$.
#
# 픽셀별 그래디언트는 체인룰로 닫힌 형태가 나옵니다:
#
# $$
# \frac{\partial \ell_p}{\partial \mu} = 2\,(f_p - g_p)\cdot f_p\cdot \frac{x_p-\mu}{s^2}
# $$

# %%
x = np.linspace(-10, 10, 401)                 # 픽셀 좌표 (1D 화면)
gt = np.exp(-(x - 3) ** 2 / (2 * 0.6 ** 2)) + np.exp(-(x + 3) ** 2 / (2 * 0.6 ** 2))  # GT: 봉우리 2개
a, s = 0.8, 2.5                               # 모델 가우시안의 진폭/폭 (넓게 한 덩어리로 덮음)


def per_pixel_grad(mu):
    f = a * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
    return 2 * (f - gt) * f * (x - mu) / s ** 2  # ∂ℓ_p/∂μ, shape [P]


g_p = per_pixel_grad(0.0)
print(f"μ=0 (두 봉우리 정중앙)  grad = Σ_p g_p = {g_p.sum():+.4f}   absgrad = Σ_p |g_p| = {np.abs(g_p).sum():.4f}")
print(f"   양(+)의 기여 합 = {g_p[g_p > 0].sum():.4f},  음(−)의 기여 합 = {g_p[g_p < 0].sum():.4f}  → 크기가 같아 상쇄")
# 출력: μ=0 (두 봉우리 정중앙)  grad = Σ_p g_p = +0.0000   absgrad = Σ_p |g_p| = 22.2615
# 출력:    양(+)의 기여 합 = 11.1307,  음(−)의 기여 합 = -11.1307  → 크기가 같아 상쇄

# %% [markdown]
# ## 2. 수치 미분으로 검증 — 일반 grad 가 정말 0 인가
#
# 닫힌 형태의 합이 autograd 결과와 같은지 중앙차분으로 확인합니다. 일반 grad 는 $\mathcal L(\mu)$ 의 기울기이므로, $\mu=0$ 이 대칭점이면 정확히 0 입니다 (정류점 — 실제 최적화라면 노이즈로 한쪽 봉우리로 미끄러지겠지만, 그 순간의 densification 통계에는 0 으로 잡힙니다).

# %%
def loss(mu):
    f = a * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
    return ((f - gt) ** 2).sum()


eps = 1e-5
fd = (loss(eps) - loss(-eps)) / (2 * eps)
print(f"중앙차분 dL/dμ|μ=0 = {fd:+.3e}   (닫힌 형태 합 {g_p.sum():+.3e})")
print(f"L(μ=0) = {loss(0):.3f},  L(μ=±1) = {loss(1):.3f} / {loss(-1):.3f}  → μ=0 은 손실의 정류점(여기서는 극대): 기울기 0 이라 Adam 은 μ 를 움직일 근거를 못 얻음")
# 출력: 중앙차분 dL/dμ|μ=0 = +0.000e+00   (닫힌 형태 합 +8.882e-16)
# 출력: L(μ=0) = 51.877,  L(μ=±1) = 50.726 / 50.726  → μ=0 은 손실의 정류점(여기서는 극대): 기울기 0 이라 Adam 은 μ 를 움직일 근거를 못 얻음

# %% [markdown]
# ## 3. μ 를 옮기며 두 지표 비교
#
# $\mu$ 를 $[-6, 6]$ 으로 쓸어 보면 일반 grad 는 대칭점 부근에서 0 을 지나가고, absgrad 는 그 자리에서도 크게 유지됩니다.
# gsplat `DefaultStrategy` 는 이 값(픽셀 단위로 스케일)을 `count` 로 나눠 `densify_grad_thresh` 와 비교해 split/clone 을 결정하므로,
# 일반 grad 를 쓰면 "정중앙에 잘못 놓인 큰 가우시안"은 임계값을 못 넘어 쪼개지지 않습니다.

# %%
mus = np.linspace(-6, 6, 241)
grad_sum = np.array([per_pixel_grad(m).sum() for m in mus])
grad_abs = np.array([np.abs(per_pixel_grad(m)).sum() for m in mus])

thresh = 0.3 * grad_abs.max()  # 토이용 임계값 (실제로는 densify_grad_thresh=0.0008, 픽셀 단위)
i0 = np.argmin(np.abs(mus))
print(f"μ=0: |grad|={abs(grad_sum[i0]):.3f} (< thresh {thresh:.3f} → 안 쪼갬)   absgrad={grad_abs[i0]:.3f} (> thresh → 쪼갬)")
print(f"|grad| 가 thresh 를 넘는 μ 구간 비율: {(np.abs(grad_sum) > thresh).mean():.0%},  absgrad 가 넘는 비율: {(grad_abs > thresh).mean():.0%}")
print(f"absgrad ≥ |grad| 가 모든 μ 에서 성립: {bool(np.all(grad_abs >= np.abs(grad_sum) - 1e-12))}")
# 출력: μ=0: |grad|=0.000 (< thresh 7.268 → 안 쪼갬)   absgrad=22.261 (> thresh → 쪼갬)
# 출력: |grad| 가 thresh 를 넘는 μ 구간 비율: 36%,  absgrad 가 넘는 비율: 100%
# 출력: absgrad ≥ |grad| 가 모든 μ 에서 성립: True

# %% [markdown]
# ## 4. 시각화
#
# 왼쪽: $\mu=0$ 에서의 장면과 픽셀별 그래디언트 $g_p$ — 왼쪽 봉우리는 $\mu$ 를 −로, 오른쪽 봉우리는 +로 당기는 기여가 거울상으로 상쇄.
# 오른쪽: $\mu$ 스윕에 따른 $|\text{grad}|$ 와 absgrad. 파선은 토이 임계값.

# %%
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.5, 0.5],
    subplot_titles=("μ=0: 장면과 픽셀별 ∂ℓ_p/∂μ", "μ 스윕: |Σ g_p| vs Σ|g_p|"),
    specs=[[{"secondary_y": True}, {}]],
)
f0 = a * np.exp(-(x - 0.0) ** 2 / (2 * s ** 2))
fig.add_trace(go.Scatter(x=x, y=gt, name="GT (봉우리 2개)", line=dict(color="#444", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=f0, name="모델 가우시안 (μ=0)", line=dict(color="#1f77b4", width=2, dash="dot")), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=g_p, name="픽셀별 g_p", marker_color=np.where(g_p > 0, "#d62728", "#2ca02c"), opacity=0.6),
              row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="intensity", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="∂ℓ_p/∂μ", row=1, col=1, secondary_y=True)
fig.update_xaxes(title_text="픽셀 좌표 x", row=1, col=1)

fig.add_trace(go.Scatter(x=mus, y=np.abs(grad_sum), name="|grad| = |Σ_p g_p|", line=dict(color="#1f77b4", width=2.5)), row=1, col=2)
fig.add_trace(go.Scatter(x=mus, y=grad_abs, name="absgrad = Σ_p |g_p|", line=dict(color="#ff7f0e", width=2.5)), row=1, col=2)
fig.add_hline(y=thresh, line=dict(color="gray", dash="dash"), annotation_text="densify thresh (toy)", row=1, col=2)
fig.add_vline(x=0, line=dict(color="black", width=1, dash="dot"), row=1, col=2)
fig.add_annotation(x=0, y=0.4, text="grad=0<br>(상쇄)", showarrow=True, arrowhead=2, ax=40, ay=-40, row=1, col=2)
fig.update_xaxes(title_text="가우시안 중심 μ", row=1, col=2)
fig.update_yaxes(title_text="크기", row=1, col=2)
fig.update_layout(height=460, width=1150, title_text="absgrad 가 '양쪽으로 당겨지는' 가우시안을 놓치지 않는 이유",
                  legend=dict(orientation="h", y=-0.2), bargap=0)
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - 일반 grad 는 $\mathcal L(\mu)$ 의 실제 기울기 → **최적화(Adam)에는 올바른 양**이지만, 대칭적으로 당겨지는 가우시안에서는 0 이 되어 "잘못 덮고 있다"는 사실을 숨김.
# - absgrad 는 픽셀별 기여의 절댓값 합 → 항상 $\ge |\text{grad}|$ 이며, 상쇄된 신호를 보존해 **densification 판정(split/clone)** 에 적합.
# - gsplat 은 `absgrad=True` 일 때 래스터라이저 backward 에서 `means2d.absgrad` 를 별도로 누적하고, `DefaultStrategy(absgrad=True)` 가 `grad2d` 통계에 이것을 사용. 파라미터 업데이트용 `.grad` 는 그대로 부호 합.
