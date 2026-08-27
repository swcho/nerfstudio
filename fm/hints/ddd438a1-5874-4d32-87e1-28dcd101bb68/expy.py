# %% [markdown]
# # 알파 블렌딩 손계산 + 2D 가우시안 스플랫 렌더링 예제
#
# 깊이 순으로 정렬된 가우시안 $i$ 에 대해
# $$\alpha_i = \sigma(\tilde\alpha_i)\exp\!\big(-\tfrac12 (p-\mu_i)^\top (\Sigma_i^{2D})^{-1}(p-\mu_i)\big),\quad
# T_i=\prod_{j<i}(1-\alpha_j),\quad
# \text{render}(p)=\sum_i c_i\alpha_i T_i,\quad
# \text{alpha}(p)=1-\prod_i(1-\alpha_i)$$

# %%
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용; 없으면 matplotlib로 대체 저장)
import os
import numpy as np

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gauss_alpha(p, mu, Sigma, alpha_logit):
    """픽셀 p(2,)에서 2D 가우시안 하나의 alpha_i."""
    d = p - mu
    maha = d @ np.linalg.inv(Sigma) @ d
    return sigmoid(alpha_logit) * np.exp(-0.5 * maha)

# %% [markdown]
# ## 1. 한 픽셀, 가우시안 4개 — 손계산 표
# 픽셀 $p=(0,0)$. 가우시안은 이미 깊이 순(앞→뒤)으로 정렬되어 있다고 가정.

# %%
p = np.array([0.0, 0.0])
G = [  # (mu, Sigma, alpha_logit, color) — 깊이 순
    (np.array([0.5, 0.0]), np.diag([1.0, 1.0]),  2.0, np.array([1.0, 0.0, 0.0])),  # 빨강, 가까움
    (np.array([0.0, 0.0]), np.diag([2.0, 0.5]), -0.5, np.array([0.0, 1.0, 0.0])),  # 초록, 중심 정확히
    (np.array([1.5, 1.0]), np.diag([1.0, 1.0]),  4.0, np.array([0.0, 0.0, 1.0])),  # 파랑, 조금 멀리
    (np.array([0.0, 0.2]), np.diag([0.3, 0.3]),  1.0, np.array([1.0, 1.0, 0.0])),  # 노랑, 가장 뒤
]

T = 1.0
render = np.zeros(3)
print(f"{'i':>2} {'σ(α~)':>6} {'exp항':>6} {'α_i':>6} {'T_i':>6} {'w_i=α_i T_i':>11}  c_i")
for i, (mu, S, al, c) in enumerate(G, 1):
    d = p - mu
    e = np.exp(-0.5 * d @ np.linalg.inv(S) @ d)
    a = sigmoid(al) * e
    w = a * T
    render += c * w
    print(f"{i:>2} {sigmoid(al):6.3f} {e:6.3f} {a:6.3f} {T:6.3f} {w:11.3f}  {c}")
    T *= (1 - a)
alpha = 1 - T
print(f"\nrender(p) = {np.round(render, 3)}")
print(f"alpha(p)  = 1 - ∏(1-α_i) = {alpha:.3f}   (= Σ w_i 확인: {1 - T:.3f})")
print(f"남은 투과율 1-alpha = {T:.3f}  → 여기에 배경색이 채워짐")
# 출력:
#  i  σ(α~)   exp항    α_i    T_i w_i=α_i T_i  c_i
#  1  0.881  0.882  0.777  1.000       0.777  [1. 0. 0.]
#  2  0.378  1.000  0.378  0.223       0.084  [0. 1. 0.]
#  3  0.982  0.197  0.193  0.139       0.027  [0. 0. 1.]
#  4  0.731  0.936  0.684  0.112       0.076  [1. 1. 0.]
#
# render(p) = [0.854 0.161 0.027]
# alpha(p)  = 1 - ∏(1-α_i) = 0.965   (= Σ w_i 확인: 0.965)
# 남은 투과율 1-alpha = 0.035  → 여기에 배경색이 채워짐

# %% [markdown]
# 가장 앞의 빨강이 $T_1=1$ 을 통째로 받아 지배적이고, 뒤로 갈수록 $T_i$ 가 줄어 기여가 작아진다.
# **순서를 뒤집으면** 같은 $\alpha_i$ 들이라도 결과가 바뀐다:

# %%
T = 1.0; render_rev = np.zeros(3)
for mu, S, al, c in reversed(G):
    a = gauss_alpha(p, mu, S, al)
    render_rev += c * a * T
    T *= (1 - a)
print("깊이 순 뒤집은 render(p) =", np.round(render_rev, 3), " alpha =", round(1 - T, 3))
# 출력: 깊이 순 뒤집은 render(p) = [0.807 0.78  0.061]  alpha = 0.965
# → alpha(p)는 순서와 무관(곱은 교환법칙)하지만 색은 크게 달라진다 (노랑이 앞으로 와 지배).

# %% [markdown]
# ## 2. 작은 2D 이미지에 가우시안 여러 개를 깊이 순 알파 블렌딩
# 모든 픽셀에 대해 같은 식을 벡터화해서 적용한다.

# %%
W = H = 64
ys, xs = np.mgrid[0:H, 0:W]
P = np.stack([xs, ys], -1).astype(float)  # (H,W,2)

splats = [  # 깊이 순 (앞→뒤): mu, Sigma, alpha_logit, color
    (np.array([22., 30.]), np.array([[60., 20.], [20., 25.]]),  1.5, np.array([1.0, 0.2, 0.2])),
    (np.array([36., 26.]), np.array([[30., 0.],  [0., 90.]]),   0.5, np.array([0.2, 0.9, 0.2])),
    (np.array([40., 40.]), np.array([[120., -40.], [-40., 60.]]), 3.0, np.array([0.2, 0.3, 1.0])),
    (np.array([28., 44.]), np.array([[40., 0.],  [0., 40.]]),   2.0, np.array([1.0, 0.9, 0.1])),
]

T_img = np.ones((H, W))
render_img = np.zeros((H, W, 3))
alpha_maps = []
for mu, S, al, c in splats:
    d = P - mu
    maha = np.einsum("hwi,ij,hwj->hw", d, np.linalg.inv(S), d)
    a = sigmoid(al) * np.exp(-0.5 * maha)          # α_i(p) 맵
    alpha_maps.append(a)
    render_img += c * (a * T_img)[..., None]        # c_i α_i T_i 누적
    T_img *= (1 - a)                                # T_{i+1} = T_i (1-α_i)
alpha_img = 1 - T_img
background = np.array([0.5, 0.5, 0.5])
rgb = np.clip(render_img + (1 - alpha_img)[..., None] * background, 0, 1)  # D7 배경 합성
print("render 범위", render_img.min().round(3), render_img.max().round(3),
      "| alpha 범위", alpha_img.min().round(3), alpha_img.max().round(3))
# 출력: render 범위 0.0 0.835 | alpha 범위 0.0 0.966

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _img(arr):  # (H,W,3) [0,1] → uint8
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

fig = make_subplots(rows=2, cols=3, subplot_titles=(
    "α_1 (빨강, 가장 앞)", "α_3 (파랑)", "T (남은 투과율 = 1-alpha)",
    "render(p) = Σ c_i α_i T_i", "alpha(p)", "rgb = render + (1-alpha)·배경(회색)"))
fig.add_trace(go.Heatmap(z=alpha_maps[0], zmin=0, zmax=1, colorscale="Reds", showscale=False), 1, 1)
fig.add_trace(go.Heatmap(z=alpha_maps[2], zmin=0, zmax=1, colorscale="Blues", showscale=False), 1, 2)
fig.add_trace(go.Heatmap(z=T_img, zmin=0, zmax=1, colorscale="Greys", showscale=False), 1, 3)
fig.add_trace(go.Image(z=_img(render_img)), 2, 1)
fig.add_trace(go.Heatmap(z=alpha_img, zmin=0, zmax=1, colorscale="Viridis", showscale=False), 2, 2)
fig.add_trace(go.Image(z=_img(rgb)), 2, 3)
for r in (1, 2):
    for c in (1, 2, 3):
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{(r-1)*3+c}", row=r, col=c)
fig.update_layout(title="깊이 순 알파 블렌딩으로 2D 가우시안 4개 렌더링 (64×64)",
                  width=1000, height=680, template="plotly_white")
_show(fig)

png = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png, scale=2)  # kaleido 필요
except Exception as e:  # kaleido 미설치 시 matplotlib로 대체 저장
    print("kaleido 저장 실패 →", type(e).__name__, "; matplotlib로 대체 저장")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(alpha_maps[0], cmap="Reds", vmin=0, vmax=1); ax[0, 0].set_title("alpha_1 (red, front)")
    ax[0, 1].imshow(alpha_maps[2], cmap="Blues", vmin=0, vmax=1); ax[0, 1].set_title("alpha_3 (blue)")
    ax[0, 2].imshow(T_img, cmap="gray", vmin=0, vmax=1); ax[0, 2].set_title("T = 1 - alpha(p)")
    ax[1, 0].imshow(render_img); ax[1, 0].set_title("render(p) = sum c_i a_i T_i")
    ax[1, 1].imshow(alpha_img, cmap="viridis", vmin=0, vmax=1); ax[1, 1].set_title("alpha(p)")
    ax[1, 2].imshow(rgb); ax[1, 2].set_title("rgb = render + (1-alpha)*bg")
    for a in ax.ravel(): a.axis("off")
    f.suptitle("Depth-ordered alpha blending of 4 2D Gaussians (64x64)")
    f.tight_layout(); f.savefig(png, dpi=120)
print("saved", png)
# 출력: saved <hint dir>/expy.png  (plotly write_image 성공)
