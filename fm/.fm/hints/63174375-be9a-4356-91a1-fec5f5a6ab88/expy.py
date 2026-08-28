# %% [markdown]
# # $l=2$ 구면 조화 함수 5개의 데카르트 표현
#
# 구면 좌표 $(\theta,\phi)$ 대신 단위 방향 벡터 $\vec n=(x,y,z)$,
# $(x,y,z)=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$ 를 쓰면
# $l=2$ 밴드의 실수 구면 조화 함수는 단순한 2차 다항식이 된다.
#
# $$
# \begin{aligned}
# y_2^{-2}&=1.092548\,xy &\quad y_2^{-1}&=1.092548\,yz \\
# y_2^{0}&=0.315392\,(3z^2-1) &\quad y_2^{1}&=1.092548\,xz \\
# y_2^{2}&=0.546274\,(x^2-y^2)
# \end{aligned}
# $$
#
# 이 스크립트는 (1) 계수가 어디서 오는지, (2) 구면좌표 정의와 수치적으로 일치하는지,
# (3) 정규직교성, (4) 각 함수의 모양을 단계적으로 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido
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
# ## 1단계: 데카르트 형태 정의 (카드의 답)
# 실제 렌더러(예: 3DGS의 `SH_C2`)에서 그대로 쓰는 상수들이다.

# %%
def sh2_cartesian(x, y, z):
    """l=2 밴드 5개를 m=-2..2 순서로 반환."""
    return np.stack([
        1.092548 * x * y,               # y_2^-2
        1.092548 * y * z,               # y_2^-1
        0.315392 * (3 * z**2 - 1),      # y_2^0
        1.092548 * x * z,               # y_2^1
        0.546274 * (x**2 - y**2),       # y_2^2
    ])


n = np.array([0.6, 0.0, 0.8])  # 단위 벡터 예시
print("|n| =", np.linalg.norm(n))
for m, v in zip(range(-2, 3), sh2_cartesian(*n)):
    print(f"y_2^{m:+d}(0.6, 0, 0.8) = {v:+.6f}")
# 출력:
# |n| = 1.0
# y_2^-2(0.6, 0, 0.8) = +0.000000
# y_2^-1(0.6, 0, 0.8) = +0.000000
# y_2^+0(0.6, 0, 0.8) = +0.290161
# y_2^+1(0.6, 0, 0.8) = +0.524423
# y_2^+2(0.6, 0, 0.8) = +0.196659

# %% [markdown]
# ## 2단계: 상수는 어디서 오나?
# 정규화 상수 $K_l^m=\sqrt{\frac{2l+1}{4\pi}\frac{(l-|m|)!}{(l+|m|)!}}$ 와
# 실수형 SH의 $\sqrt2$ 인자, 그리고 버금 르장드르 다항식 $P_2^{|m|}(\cos\theta)$ 의 앞 계수를 곱하면 된다.
#
# - $P_2^0(z)=\tfrac12(3z^2-1)$ → $0.315392 = \tfrac12\sqrt{\tfrac{5}{4\pi}}$
# - $P_2^1(z)=-3z\sqrt{1-z^2}$ → $1.092548=\sqrt2\cdot 3\cdot\sqrt{\tfrac{5}{4\pi}\tfrac{1}{6}}=\tfrac12\sqrt{\tfrac{15}{\pi}}$
# - $P_2^2(z)=3(1-z^2)$ → $0.546274=\sqrt2\cdot 3\cdot\sqrt{\tfrac{5}{4\pi}\tfrac{1}{24}}=\tfrac14\sqrt{\tfrac{15}{\pi}}$

# %%
from math import pi, sqrt, factorial


def K(l, m):
    return sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))


print("0.315392 ?=", 0.5 * K(2, 0))
print("1.092548 ?=", sqrt(2) * 3 * K(2, 1), "=", 0.5 * sqrt(15 / pi))
print("0.546274 ?=", sqrt(2) * 3 * K(2, 2), "=", 0.25 * sqrt(15 / pi))
# 출력:
# 0.315392 ?= 0.31539156525252005
# 1.092548 ?= 1.0925484305920794 = 1.0925484305920792
# 0.546274 ?= 0.5462742152960397 = 0.5462742152960396

# %% [markdown]
# ## 3단계: 구면좌표 정의와 수치적으로 일치하는지 확인
# 원문의 `SH(l, m, theta, phi)` 정의(m=0: $K P$, m>0: $\sqrt2 K\cos(m\phi)P$, m<0: $\sqrt2 K\sin(|m|\phi)P$)를
# 그대로 구현하고, 데카르트 형태와 무작위 방향에서 비교한다.
#
# 주의: 원문 재귀식의 $P_l^m$ 은 **콘던–쇼틀리 위상** $(-1)^m$ 을 포함한다
# ($P_2^1=-3z\sqrt{1-z^2}$). 표로 정리된 데카르트 형태(그리고 3DGS 등의 `SH_C2`)는 이 부호를 뺀 것이라
# $m=\pm1$ 에서 부호만 반대로 나온다. 그래프 렌더링/투영 재구성에서는 기저와 계수가 같은 부호 규약을 쓰면 되므로 결과에 영향이 없다.

# %%
def P(l, m, x):
    """버금 르장드르 다항식 P_l^m(x), 원문 코드와 같은 재귀."""
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def sh_spherical(l, m, theta, phi):
    if m == 0:
        return K(l, 0) * P(l, 0, np.cos(theta))
    if m > 0:
        return sqrt(2) * K(l, m) * np.cos(m * phi) * P(l, m, np.cos(theta))
    return sqrt(2) * K(l, -m) * np.sin(-m * phi) * P(l, -m, np.cos(theta))


rng = np.random.default_rng(0)
theta = rng.uniform(0, pi, 1000)
phi = rng.uniform(0, 2 * pi, 1000)
x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

cart = sh2_cartesian(x, y, z)
sph = np.stack([sh_spherical(2, m, theta, phi) for m in range(-2, 3)])
for i, m in enumerate(range(-2, 3)):
    d_raw = np.abs(cart[i] - sph[i]).max()
    d_fix = np.abs(cart[i] - (-1) ** abs(m) * sph[i]).max()   # 콘던-쇼틀리 위상 제거
    print(f"m={m:+d}: max|cart - sph| = {d_raw:.2e},  위상 (-1)^|m| 제거 후 = {d_fix:.2e}")
# 출력:
# m=-2: max|cart - sph| = 2.15e-07,  위상 (-1)^|m| 제거 후 = 2.15e-07
# m=-1: max|cart - sph| = 1.09e+00,  위상 (-1)^|m| 제거 후 = 2.14e-07
# m=+0: max|cart - sph| = 8.69e-07,  위상 (-1)^|m| 제거 후 = 8.69e-07
# m=+1: max|cart - sph| = 1.09e+00,  위상 (-1)^|m| 제거 후 = 2.15e-07
# m=+2: max|cart - sph| = 2.15e-07,  위상 (-1)^|m| 제거 후 = 2.15e-07
# → m=±1은 부호만 다르고(콘던-쇼틀리 위상), 위상을 맞추면 전부 1e-6 이내로 일치. 잔차는 소수 6자리 반올림 상수 탓.

# %% [markdown]
# ## 4단계: 정규직교성 확인
# 구면 위 몬테카를로 적분으로 $\int y_i y_j\, d\Omega \approx 4\pi\cdot\mathrm{mean}(y_i y_j)$ 가
# 단위행렬에 가까운지 본다. 이 성질 덕분에 투영 계수를 단순 적분으로 얻을 수 있다.

# %%
N = 200_000
v = rng.normal(size=(3, N))
v /= np.linalg.norm(v, axis=0)          # 균일한 구면 샘플
Y = sh2_cartesian(*v)                    # (5, N)
G = 4 * pi * (Y @ Y.T) / N
np.set_printoptions(precision=3, suppress=True)
print(G)
# 출력:
# [[ 1.004 -0.002  0.002 -0.007 -0.001]
#  [-0.002  1.     0.001  0.001 -0.001]
#  [ 0.002  0.001  0.998 -0.001  0.002]
#  [-0.007  0.001 -0.001  1.001 -0.002]
#  [-0.001 -0.001  0.002 -0.002  0.998]]
# (몬테카를로 오차 ~1e-3 범위에서 단위행렬)

# %% [markdown]
# ## 5단계: 다섯 함수의 모양
# 반지름을 $|y_2^m(\vec n)|$ 로 두고 부호를 색으로 표시한 전형적인 "로브" 그림.
# - $xy, yz, xz, x^2-y^2$: 네 개의 로브 (클로버 모양), 축 위치만 다름
# - $3z^2-1$: z축 방향 두 로브 + 적도 둘레의 도넛

# %%
th = np.linspace(0, pi, 60)
ph = np.linspace(0, 2 * pi, 120)
TH, PH = np.meshgrid(th, ph)
X, Yc, Z = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)
vals = sh2_cartesian(X, Yc, Z)

titles = ["y₂⁻² = 1.092548·xy", "y₂⁻¹ = 1.092548·yz", "y₂⁰ = 0.315392(3z²−1)",
          "y₂¹ = 1.092548·xz", "y₂² = 0.546274(x²−y²)"]
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "surface"}] * 5],
                    subplot_titles=titles, horizontal_spacing=0.01)
for i in range(5):
    r = np.abs(vals[i])
    fig.add_trace(go.Surface(x=r * X, y=r * Yc, z=r * Z, surfacecolor=np.sign(vals[i]),
                             colorscale=[[0, "#3b6fd6"], [1, "#d64b3b"]], cmin=-1, cmax=1,
                             showscale=False), row=1, col=i + 1)
lim = 0.7
scene = dict(xaxis=dict(range=[-lim, lim], visible=False), yaxis=dict(range=[-lim, lim], visible=False),
             zaxis=dict(range=[-lim, lim], visible=False), aspectmode="cube",
             camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)))
fig.update_layout(height=340, width=1500, margin=dict(l=0, r=0, t=50, b=0),
                  title_text="l=2 구면 조화 함수 (빨강: +, 파랑: −, 반지름 = |y|)",
                  **{f"scene{i}" if i > 1 else "scene": scene for i in range(1, 6)})
_show(fig)
import os
fig.write_image(os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "expy.png"), scale=1.5)
print("saved expy.png")
# 출력:
# saved expy.png

# %% [markdown]
# ## 정리
# - $l=2$ 밴드는 $x,y,z$의 **2차 동차식** 5개: $xy,\ yz,\ 3z^2-1,\ xz,\ x^2-y^2$.
# - 계수 $1.092548=\tfrac12\sqrt{15/\pi}$, $0.546274=\tfrac14\sqrt{15/\pi}$, $0.315392=\tfrac14\sqrt{5/\pi}$
#   는 $K_l^m$, $\sqrt2$, 르장드르 계수의 곱이며, 구면 위에서 정규직교가 되게 맞춘 값이다.
# - 표의 데카르트 형태는 콘던–쇼틀리 위상 $(-1)^m$ 을 뺀 규약(그래픽스 관행)이라 $m=\pm1$ 부호가 교과서식 $P_l^m$ 과 다를 수 있다.
# - 구현에서는 $\theta,\phi$ 를 거치지 않고 단위 벡터 $\vec n$ 을 바로 다항식에 넣기만 하면 된다.
