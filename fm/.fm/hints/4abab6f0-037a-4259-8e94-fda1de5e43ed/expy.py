# %% [markdown]
# # 최적화된 Irradiance 렌더링 식의 상수 $c_1 \sim c_5$ 는 어디서 오는가?
#
# Ramamoorthi & Hanrahan (2001) 의 식
#
# $$E(n)=c_1L_{22}(x^2-y^2)+c_3L_{20}z^2+c_4L_{00}-c_5L_{20}+2c_1(L_{2-2}xy+L_{21}xz+L_{2-1}yz)+2c_2(L_{11}x+L_{1-1}y+L_{10}z)$$
#
# 의 상수 $c_1=0.429043,\ c_2=0.511664,\ c_3=0.743125,\ c_4=0.886227,\ c_5=0.247708$ 는
# 외워야 하는 마법의 숫자가 아니라 **두 가지 상수의 곱**이다.
#
# 1. 클램프된 코사인 $\max(\cos\theta,0)$ 의 SH 계수 $\hat{A}_l=\sqrt{\tfrac{4\pi}{2l+1}}A_l$
#    ($\hat A_0=\pi,\ \hat A_1=2\pi/3,\ \hat A_2=\pi/4$)
# 2. 실수 SH 기저 $y_l^m$ 의 정규화 상수 (셰이더 `ShFunctionL2` 의 0.282095, 0.488603, 1.092548, 0.315392, 0.546274)
#
# 이 스크립트는 그 사실을 수치로 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido
import os
import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

# %% [markdown]
# ## 1. $\hat{A}_l$ 계산
#
# $A_l=\int_0^{\pi}\max(\cos\theta,0)\,y_l^0(\theta,0)\,\sin\theta\,d\theta$ 를 수치 적분(방위각 $2\pi$ 포함)하고
# $\hat A_l=\sqrt{4\pi/(2l+1)}A_l$ 로 변환한다. 해석해는 $\hat A_0=\pi,\ \hat A_1=\frac{2\pi}{3},\ \hat A_2=\frac{\pi}{4}$.

# %%
# 셰이더 ShFunctionL2 와 동일한 정규화 상수
K00, K1, K2xy, K20, K22 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274

theta = np.linspace(0, np.pi, 200001)
z = np.cos(theta)
clamped = np.maximum(z, 0.0)
w = np.sin(theta) * 2 * np.pi          # dΩ = sinθ dθ dφ, φ 적분은 2π (m=0 이므로 φ 무관)

def trapz(f):
    return np.trapezoid(f, theta) if hasattr(np, "trapezoid") else np.trapz(f, theta)

A0 = trapz(clamped * K00 * w)
A1 = trapz(clamped * K1 * z * w)
A2 = trapz(clamped * K20 * (3 * z**2 - 1) * w)
Ahat = [np.sqrt(4 * np.pi / (2 * l + 1)) * A for l, A in enumerate((A0, A1, A2))]
for l, (a, ref) in enumerate(zip(Ahat, (np.pi, 2 * np.pi / 3, np.pi / 4))):
    print(f"A_hat_{l} = {a:.5f}   (해석해 {ref:.5f})")
# 출력:
# A_hat_0 = 3.14159   (해석해 3.14159)
# A_hat_1 = 2.09440   (해석해 2.09440)
# A_hat_2 = 0.78540   (해석해 0.78540)

# %% [markdown]
# ## 2. $c_1\sim c_5$ 유도
#
# $E(n)=\sum_{lm}\hat A_l L_{lm}\,y_l^m(n)$ 에 $y_l^m$ 의 다항식 형태를 대입해 항별로 묶으면
#
# | 상수 | 유도 | 관련 항 |
# |---|---|---|
# | $c_4=\hat A_0\cdot 0.282095$ | $\pi\cdot0.282095$ | $L_{00}$ |
# | $c_2=\tfrac12\hat A_1\cdot 0.488603$ | 식에는 $2c_2$ 로 등장 | $L_{1m}\,(x,y,z)$ |
# | $c_1=\hat A_2\cdot 0.546274=\tfrac12\hat A_2\cdot1.092548$ | 두 방식이 일치 | $L_{22}(x^2-y^2)$, $2c_1\cdot xy,xz,yz$ |
# | $c_3=\hat A_2\cdot 3\cdot0.315392$ | $y_2^0=0.315392(3z^2-1)$ 의 $3z^2$ | $L_{20}z^2$ |
# | $c_5=\hat A_2\cdot 0.315392$ | 같은 $y_2^0$ 의 $-1$ | $-L_{20}$ |

# %%
A0h, A1h, A2h = np.pi, 2 * np.pi / 3, np.pi / 4
c = {
    "c1": A2h * K22,
    "c2": 0.5 * A1h * K1,
    "c3": A2h * 3 * K20,
    "c4": A0h * K00,
    "c5": A2h * K20,
}
paper = {"c1": 0.429043, "c2": 0.511664, "c3": 0.743125, "c4": 0.886227, "c5": 0.247708}
for k in c:
    print(f"{k} = {c[k]:.6f}   (논문/셰이더 {paper[k]:.6f}, 차이 {abs(c[k]-paper[k]):.1e})")
print("c1 via 0.5*A2h*1.092548 =", round(0.5 * A2h * K2xy, 6))
# 출력:
# c1 = 0.429043   (논문/셰이더 0.429043, 차이 4.0e-07)
# c2 = 0.511664   (논문/셰이더 0.511664, 차이 1.3e-07)
# c3 = 0.743125   (논문/셰이더 0.743125, 차이 1.1e-07)
# c4 = 0.886228   (논문/셰이더 0.886227, 차이 5.8e-07)  ← 논문 값은 6자리 반올림
# c5 = 0.247708   (논문/셰이더 0.247708, 차이 3.0e-07)
# c1 via 0.5*A2h*1.092548 = 0.429043

# %% [markdown]
# ## 3. 최적화 식 vs. 일반 SH 합 검증
#
# 임의의 $L_{lm}$ (9개) 와 임의의 법선 $n$ 에 대해
# 셰이더 `ImageBasedLight` 형태의 식과 $\sum_{lm}\hat A_l L_{lm} y_l^m(n)$ 이 일치해야 한다.

# %%
def sh_l2(v):
    x, y, z = v
    return np.array([K00, K1 * y, K1 * z, K1 * x,
                     K2xy * x * y, K2xy * y * z, K20 * (3 * z * z - 1), K2xy * x * z, K22 * (x * x - y * y)])

def irradiance_generic(L, n):
    Ah = np.array([A0h, A1h, A1h, A1h, A2h, A2h, A2h, A2h, A2h])
    return np.sum(Ah * L * sh_l2(n))

def irradiance_optimized(L, n):
    l00, l1_1, l10, l11, l2_2, l2_1, l20, l21, l22 = L
    x, y, z = n
    c1, c2, c3, c4, c5 = paper["c1"], paper["c2"], paper["c3"], paper["c4"], paper["c5"]
    return (c1 * l22 * (x * x - y * y) + c3 * l20 * z * z + c4 * l00 - c5 * l20
            + 2 * c1 * (l2_2 * x * y + l21 * x * z + l2_1 * y * z)
            + 2 * c2 * (l11 * x + l1_1 * y + l10 * z))

rng = np.random.default_rng(0)
max_err = 0.0
for _ in range(1000):
    L = rng.normal(size=9)
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    max_err = max(max_err, abs(irradiance_generic(L, n) - irradiance_optimized(L, n)))
print(f"1000개 랜덤 (L, n) 에 대한 최대 오차: {max_err:.2e}")
# 출력:
# 1000개 랜덤 (L, n) 에 대한 최대 오차: 2.55e-06

# %% [markdown]
# ## 4. 시각화
#
# 왼쪽: 각 상수가 어떤 $\hat A_l$ 에서 오는지 (색은 $l$).
# 오른쪽: $\hat A_l$ 의 빠른 감쇠 — $l\le2$ 만 쓰는 이유.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

names = ["c1", "c2", "c3", "c4", "c5"]
band = {"c1": 2, "c2": 1, "c3": 2, "c4": 0, "c5": 2}
colors = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}

fig = make_subplots(rows=1, cols=2, subplot_titles=("c₁~c₅ (색: 유래한 밴드 l)", "Â_l 감쇠 (l=0..6)"))
fig.add_trace(go.Bar(x=names, y=[c[k] for k in names],
                     marker_color=[colors[band[k]] for k in names],
                     text=[f"{c[k]:.6f}" for k in names], textposition="outside", showlegend=False), row=1, col=1)

# Â_l 일반 항 (l 짝수, l≥2): 2π (-1)^{l/2-1} (l-2)!... 대신 수치 적분으로 계산
def Ahat_numeric(l):
    from numpy.polynomial import legendre
    P = legendre.Legendre.basis(l)(z)                     # P_l(cosθ)
    y_l0 = np.sqrt((2 * l + 1) / (4 * np.pi)) * P         # y_l^0
    return np.sqrt(4 * np.pi / (2 * l + 1)) * trapz(clamped * y_l0 * w)

ls = list(range(7))
Ah_all = [Ahat_numeric(l) for l in ls]
print("A_hat_l, l=0..6:", [round(float(a), 4) for a in Ah_all])
# 출력:
# A_hat_l, l=0..6: [3.1416, 2.0944, 0.7854, -0.0, -0.1309, -0.0, 0.0491]
fig.add_trace(go.Scatter(x=ls, y=Ah_all, mode="lines+markers",
                         marker=dict(color=[colors.get(l, "#999") for l in ls], size=10), showlegend=False), row=1, col=2)
fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
fig.update_xaxes(title_text="l", row=1, col=2)
fig.update_layout(height=420, width=900, title_text="Irradiance SH 상수 c₁~c₅ = Â_l × SH 정규화 상수")
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"))
print("saved expy.png")
# 출력:
# saved expy.png
