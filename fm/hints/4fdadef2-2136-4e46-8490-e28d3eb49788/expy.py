# %% [markdown]
# # SSIM 직접 계산해 보기 (numpy)
# splatfacto가 쓰는 설정($11\times11$ 가우시안 윈도우, $\sigma=1.5$, data_range 1,
# $C_1=0.01^2,\ C_2=0.03^2$)을 numpy로 재현하고, **L1이 같지만 SSIM은 다른** 두 왜곡을 비교합니다.
#
# $$\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$

# %%
# 필요 패키지: numpy, scipy, plotly, kaleido
import numpy as np
from scipy.signal import convolve2d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. 11×11 가우시안 윈도우 (σ = 1.5)
# $w_{ij} \propto \exp\!\left(-\frac{i^2+j^2}{2\sigma^2}\right)$, 합이 1이 되도록 정규화.

# %%
def gaussian_window(size=11, sigma=1.5):
    r = np.arange(size) - size // 2
    g = np.exp(-(r ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)

W = gaussian_window()
print("shape:", W.shape, " sum:", W.sum().round(6))
print("중심 가중치:", W[5, 5].round(4), " 모서리 가중치:", W[0, 0].round(8))
# 출력: shape: (11, 11)  sum: 1.0
# 출력: 중심 가중치: 0.0708  모서리 가중치: 1.06e-06

# %% [markdown]
# ## 2. 국소 통계량 → SSIM 맵
# 합성곱으로 $\mu_x,\mu_y,E[x^2],E[y^2],E[xy]$를 구하고
# $\sigma_x^2=E[x^2]-\mu_x^2,\ \sigma_{xy}=E[xy]-\mu_x\mu_y$. torchmetrics 기본과 같이 `valid` 영역만 평균합니다.

# %%
def ssim_map(x, y, K1=0.01, K2=0.03, data_range=1.0, win=W):
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    f = lambda a: convolve2d(a, win, mode="valid")
    mu_x, mu_y = f(x), f(y)
    sxx = f(x * x) - mu_x ** 2
    syy = f(y * y) - mu_y ** 2
    sxy = f(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
    return num / den

def ssim(x, y):
    return ssim_map(x, y).mean()

# %% [markdown]
# ## 3. 합성 이미지: 체커보드 + 기울기, 두 종류의 왜곡
# - **밝기 이동**: 전체에 상수 $\delta$를 더함 → 구조는 그대로
# - **블러**: 가우시안 흐림 → 구조(대비)가 깨짐. L1이 밝기 이동과 같도록 $\delta$를 맞춤.

# %%
rng = np.random.default_rng(0)
H = 96
yy, xx = np.mgrid[0:H, 0:H]
img = 0.5 * (((xx // 12 + yy // 12) % 2)) + 0.5 * (xx / H)
img = 0.1 + 0.75 * np.clip(img, 0, 1)  # [0.1, 0.85] — 밝기 이동 시 클리핑 방지

blur_win = gaussian_window(9, 2.0)
blurred = convolve2d(img, blur_win, mode="same", boundary="symm")
l1_blur = np.abs(blurred - img).mean()

shifted = np.clip(img + l1_blur, 0, 1)  # 같은 L1이 되도록 상수 이동
l1_shift = np.abs(shifted - img).mean()

print(f"L1(블러)      = {l1_blur:.4f}   SSIM = {ssim(img, blurred):.4f}")
print(f"L1(밝기 이동) = {l1_shift:.4f}   SSIM = {ssim(img, shifted):.4f}")
print(f"SSIM(자기 자신) = {ssim(img, img):.4f}")
# 출력: L1(블러)      = 0.0720   SSIM = 0.6151
# 출력: L1(밝기 이동) = 0.0720   SSIM = 0.9846
# 출력: SSIM(자기 자신) = 1.0000

# %% [markdown]
# 같은 L1인데도 블러는 SSIM이 크게 떨어지고, 밝기 이동은 거의 1에 머뭅니다.
# splatfacto 손실 $0.8\,\mathcal L_1 + 0.2\,(1-\mathrm{SSIM})$ 에서 두 경우가 어떻게 갈리는지 확인합니다.

# %%
lam = 0.2
for name, d in [("블러", blurred), ("밝기 이동", shifted)]:
    L1 = np.abs(d - img).mean(); s = ssim(img, d)
    print(f"{name:6s}: loss = 0.8·{L1:.4f} + 0.2·{1 - s:.4f} = {(1 - lam) * L1 + lam * (1 - s):.4f}")
# 출력: 블러    : loss = 0.8·0.0720 + 0.2·0.3849 = 0.1346
# 출력: 밝기 이동 : loss = 0.8·0.0720 + 0.2·0.0154 = 0.0607

# %% [markdown]
# ## 4. SSIM 맵 시각화
# 블러 맵은 체커보드 **경계**에서 값이 뚝 떨어지고(구조 손실), 밝기 이동 맵은 전체가 거의 1입니다.

# %%
m_blur, m_shift = ssim_map(img, blurred), ssim_map(img, shifted)
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.04,
                    subplot_titles=("원본", "블러", f"SSIM 맵(블러) 평균 {m_blur.mean():.3f}",
                                    f"SSIM 맵(밝기 이동) 평균 {m_shift.mean():.3f}"))
fig.add_trace(go.Heatmap(z=img, colorscale="gray", zmin=0, zmax=1, showscale=False), 1, 1)
fig.add_trace(go.Heatmap(z=blurred, colorscale="gray", zmin=0, zmax=1, showscale=False), 1, 2)
fig.add_trace(go.Heatmap(z=m_blur, colorscale="Viridis", zmin=0, zmax=1, showscale=False), 1, 3)
fig.add_trace(go.Heatmap(z=m_shift, colorscale="Viridis", zmin=0, zmax=1,
                         colorbar=dict(title="SSIM", x=1.02)), 1, 4)
for c in range(1, 5):
    fig.update_yaxes(autorange="reversed", scaleanchor=f"x{c if c > 1 else ''}", showticklabels=False, row=1, col=c)
    fig.update_xaxes(showticklabels=False, row=1, col=c)
fig.update_layout(width=1400, height=400, title="같은 L1, 다른 SSIM: 블러 vs 밝기 이동")
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved expy.png")
# 출력: saved expy.png
