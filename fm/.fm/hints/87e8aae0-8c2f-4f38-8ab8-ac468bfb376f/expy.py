# %% [markdown]
# # $l=0$, $l=1$ 구면 조화 함수를 데카르트 좌표로
#
# 구면 좌표 $(\theta,\phi)$로 정의된 실수 구면 조화 함수를
# $(x,y,z)=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$ 로 바꾸면
#
# $$y_0^0=0.282095,\quad y_1^{-1}=0.488603\,y,\quad y_1^0=0.488603\,z,\quad y_1^1=0.488603\,x$$
#
# 가 됨을 (1) 상수 유도, (2) 정의식 vs 데카르트식 수치 비교, (3) 직교정규성 확인, (4) 시각화로 살펴본다.

# %%
# 필요 패키지: numpy, plotly, kaleido
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

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

# %% [markdown]
# ## 1. 상수 0.282095, 0.488603 은 어디서 오나
#
# $K_l^m=\sqrt{\dfrac{2l+1}{4\pi}\dfrac{(l-|m|)!}{(l+|m|)!}}$ 이고,
# $m\neq0$ 이면 $\sqrt2$ 가 추가로 곱해진다. $l=1,|m|=1$에서 $P_1^1(\cos\theta)=-\sin\theta$ 의 부호는
# 관례(Condon–Shortley)에 따라 그래픽스 표에서는 양으로 잡는다.

# %%
from math import factorial, pi, sqrt

def K(l, m):
    m = abs(m)
    return sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))

print("y_0^0 상수  K(0,0)        =", round(K(0, 0), 6), " (= 1/(2*sqrt(pi)))")
print("y_1^0 상수  K(1,0)        =", round(K(1, 0), 6), " (= sqrt(3/(4pi)))")
print("y_1^±1 상수 sqrt2*K(1,1)  =", round(sqrt(2) * K(1, 1), 6), " (P_1^1 = sin(theta) 이므로 x,y 계수)")
# 출력: y_0^0 상수  K(0,0)        = 0.282095  (= 1/(2*sqrt(pi)))
# 출력: y_1^0 상수  K(1,0)        = 0.488603  (= sqrt(3/(4pi)))
# 출력: y_1^±1 상수 sqrt2*K(1,1)  = 0.488603  (P_1^1 = sin(theta) 이므로 x,y 계수)

# %% [markdown]
# ## 2. 구면 좌표 정의식 vs 데카르트식 — 임의의 방향에서 같은가
#
# 정의식(원문의 `SH(l,m,theta,phi)`):
# - $m=0$: $K_l^0 P_l^0(\cos\theta)$
# - $m>0$: $\sqrt2 K_l^m \cos(m\phi) P_l^m(\cos\theta)$
# - $m<0$: $\sqrt2 K_l^{|m|} \sin(|m|\phi) P_l^{|m|}(\cos\theta)$
#
# $l\le1$ 에서 필요한 르장드르 값: $P_0^0=1,\;P_1^0=\cos\theta,\;P_1^1=\sin\theta$.

# %%
def sh_spherical(l, m, theta, phi):
    ct, st = np.cos(theta), np.sin(theta)
    P = {(0, 0): np.ones_like(ct), (1, 0): ct, (1, 1): st}[(l, abs(m))]
    if m == 0:
        return K(l, 0) * P
    if m > 0:
        return sqrt(2) * K(l, m) * np.cos(m * phi) * P
    return sqrt(2) * K(l, -m) * np.sin(-m * phi) * P

def sh_cartesian(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return {
        (0, 0): 0.282095 * np.ones_like(x),
        (1, -1): 0.488603 * y,
        (1, 0): 0.488603 * z,
        (1, 1): 0.488603 * x,
    }

rng = np.random.default_rng(0)
theta = rng.uniform(0, np.pi, 1000)
phi = rng.uniform(0, 2 * np.pi, 1000)
n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
cart = sh_cartesian(n)
for (l, m), v in cart.items():
    err = np.abs(v - sh_spherical(l, m, theta, phi)).max()
    print(f"y_{l}^{m:>2}: 최대 오차 = {err:.2e}")
# 출력: y_0^ 0: 최대 오차 = 2.08e-07
# 출력: y_1^-1: 최대 오차 = 4.86e-07
# 출력: y_1^ 0: 최대 오차 = 4.88e-07
# 출력: y_1^ 1: 최대 오차 = 4.87e-07
# (오차 ~1e-7 은 표의 상수를 소수 6자리로 반올림한 탓)

# %% [markdown]
# ## 3. 직교정규성 — 구면 위에서 $\int y_i y_j\,dA=\delta_{ij}$
#
# 구면 위 균일 샘플(몬테카를로)로 $4\pi\cdot\mathrm{mean}(y_i y_j)$ 를 계산하면 단위행렬에 가까워야 한다.
# 이것이 $y_0^0$ 의 상수가 $1/\sqrt{4\pi}$, $y_1^m$ 의 상수가 $\sqrt{3/(4\pi)}$ 인 이유다.

# %%
N = 400_000
v = rng.normal(size=(N, 3))
v /= np.linalg.norm(v, axis=1, keepdims=True)  # 구면 균일 분포
Y = np.stack(list(sh_cartesian(v).values()), 0)  # (4, N)
G = 4 * np.pi * (Y @ Y.T) / N
np.set_printoptions(precision=3, suppress=True)
print("그램 행렬 (순서: y00, y1-1, y10, y11)\n", G)
# 출력: 그램 행렬 (순서: y00, y1-1, y10, y11)
# 출력:  [[ 1.     0.001 -0.001  0.003]
# 출력:   [ 0.001  0.998 -0.     0.   ]
# 출력:   [-0.001 -0.     1.002 -0.001]
# 출력:   [ 0.003  0.    -0.001  1.   ]]

# %% [markdown]
# ## 4. 시각화 — 구 표면 위 함수값
#
# 반지름 1인 구 표면에 $y_l^m$ 값을 색으로 칠한다.
# $y_0^0$ 는 균일한 단색, $y_1^{-1},y_1^0,y_1^1$ 은 각각 $y,z,x$ 축 방향으로 양(+)/음(-) 반구가 갈린다.

# %%
th, ph = np.meshgrid(np.linspace(0, np.pi, 60), np.linspace(0, 2 * np.pi, 120), indexing="ij")
xs, ys, zs = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
grid = np.stack([xs, ys, zs], -1)
vals = sh_cartesian(grid)
titles = ["y_0^0 = 0.282095", "y_1^-1 = 0.488603 y", "y_1^0 = 0.488603 z", "y_1^1 = 0.488603 x"]

fig = make_subplots(rows=1, cols=4, specs=[[{"type": "surface"}] * 4], subplot_titles=titles,
                    horizontal_spacing=0.01)
for i, ((l, m), c) in enumerate(vals.items(), start=1):
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, surfacecolor=c, cmin=-0.5, cmax=0.5,
                             colorscale="RdBu_r", showscale=(i == 4),
                             colorbar=dict(title="값", len=0.6)), row=1, col=i)
scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube",
             camera=dict(eye=dict(x=1.4, y=1.2, z=0.9)))
fig.update_layout(height=380, width=1500, margin=dict(l=0, r=0, t=50, b=0),
                  title_text="l=0, l=1 실수 구면 조화 함수 (구 표면 색 = 함수값)",
                  **{f"scene{i if i > 1 else ''}": scene for i in range(1, 5)})
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=1.5)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
# - $l=0$: 방향에 무관한 상수 $1/(2\sqrt\pi)=0.282095$ → "평균" 성분.
# - $l=1$: $\sqrt{3/(4\pi)}=0.488603$ 에 각 축 성분 $(y,z,x)$ 를 곱한 것 → "어느 축으로 기울었나" 성분.
# - 두 상수는 구면 위에서 제곱 적분이 1이 되도록(정규화) 정해진 값이다.
