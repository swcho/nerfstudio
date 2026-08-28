# %% [markdown]
# # 미리 계산된 $\hat{A_l}$ 값 — 직접 계산해 보기
#
# Irradiance $E$ 의 SH 계수는 Radiance 계수 $L_{lm}$ 과 클램프된 코사인 $\max(\cos\theta,0)$ 의
# SH 계수 $A_l$ 의 곱으로 얻어진다.
#
# $$E_{lm}=\sqrt{\frac{4\pi}{2l+1}}\,A_l\,L_{lm} = \hat{A_l}\,L_{lm},\qquad
# \hat{A_l}=\sqrt{\frac{4\pi}{2l+1}}\,A_l$$
#
# 여기서
# $$A_l = 2\pi\int_0^{\pi}\max(\cos\theta,0)\,Y_l^0(\theta)\,\sin\theta\,d\theta$$
#
# 글에서 주어진 상수는 $\hat{A_0}=3.1415,\ \hat{A_1}=2.0943,\ \hat{A_2}=0.7853,\ \hat{A_3}=0,\
# \hat{A_4}=-0.1309,\ \hat{A_5}=0,\ \hat{A_6}=0.0490$ 이다.
# 이 값을 수치 적분으로 재현하고, 왜 홀수 차수가 0인지 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido
import math
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
# ## 1. $Y_l^0(\theta)$ — 띠(zonal) 조화 함수
#
# $m=0$ 인 구면 조화 함수는 $\phi$ 에 의존하지 않고 Legendre 다항식 $P_l(\cos\theta)$ 로만 표현된다.
#
# $$Y_l^0(\theta)=\sqrt{\frac{2l+1}{4\pi}}\,P_l(\cos\theta)$$
#
# $\cos\theta$ 는 천정각(zenith)에만 의존하므로 $m\neq0$ 인 성분은 모두 0 → $A_l$ 만 필요하다.

# %%
def Y_l0(l, theta):
    P = np.polynomial.legendre.Legendre.basis(l)
    return math.sqrt((2 * l + 1) / (4 * math.pi)) * P(np.cos(theta))


theta = np.linspace(0, math.pi, 200001)
for l in range(3):
    print(f"Y_{l}^0(theta=0) = {Y_l0(l, 0.0):.4f}")
# 출력: Y_0^0(theta=0) = 0.2821
# 출력: Y_1^0(theta=0) = 0.4886
# 출력: Y_2^0(theta=0) = 0.6308

# %% [markdown]
# ## 2. $A_l$ 과 $\hat{A_l}$ 수치 적분
#
# 적분 구간을 $\theta\in[0,\pi]$ 전체(구 전체)로 두고, 피적분 함수에 $\max(\cos\theta,0)$ 를 넣어
# 아래 반구가 자연스럽게 0이 되게 한다.

# %%
def A_hat(l):
    clamped_cos = np.maximum(np.cos(theta), 0.0)
    integrand = clamped_cos * Y_l0(l, theta) * np.sin(theta)
    A_l = 2 * math.pi * np.trapezoid(integrand, theta)
    return math.sqrt(4 * math.pi / (2 * l + 1)) * A_l


A_hat_vals = [A_hat(l) for l in range(7)]
for l, v in enumerate(A_hat_vals):
    print(f"A_hat_{l} = {v: .4f}")
# 출력: A_hat_0 =  3.1416
# 출력: A_hat_1 =  2.0944
# 출력: A_hat_2 =  0.7854
# 출력: A_hat_3 = -0.0000
# 출력: A_hat_4 = -0.1309
# 출력: A_hat_5 = -0.0000
# 출력: A_hat_6 =  0.0491

# %% [markdown]
# 글의 값(3.1415, 2.0943, 0.7853, 0, −0.1309, 0, 0.0490)과 일치한다(반올림 대신 절삭된 표기).
#
# ## 3. 닫힌 형식 (Ramamoorthi & Hanrahan 2001)
#
# $$\hat{A_0}=\pi,\qquad \hat{A_1}=\frac{2\pi}{3},\qquad
# \hat{A_l}=2\pi\,\frac{(-1)^{l/2-1}}{(l+2)(l-1)}\left[\frac{l!}{2^l\,((l/2)!)^2}\right]\ (l\ \text{짝수},\ l\ge2),\qquad
# \hat{A_l}=0\ (l\ \text{홀수},\ l\ge3)$$
#
# 즉 $\hat{A_2}=\pi/4,\ \hat{A_4}=-\pi/24,\ \hat{A_6}=\pi/64$ 이다.

# %%
def A_hat_closed(l):
    if l == 0:
        return math.pi
    if l == 1:
        return 2 * math.pi / 3
    if l % 2 == 1:
        return 0.0
    h = l // 2
    return 2 * math.pi * (-1) ** (h - 1) / ((l + 2) * (l - 1)) * math.factorial(l) / (2**l * math.factorial(h) ** 2)


names = ["pi", "2pi/3", "pi/4", "0", "-pi/24", "0", "pi/64"]
for l in range(7):
    print(f"l={l}: closed={A_hat_closed(l): .4f} ({names[l]:>6})  numeric={A_hat_vals[l]: .4f}")
# 출력: l=0: closed= 3.1416 (    pi)  numeric= 3.1416
# 출력: l=1: closed= 2.0944 ( 2pi/3)  numeric= 2.0944
# 출력: l=2: closed= 0.7854 (  pi/4)  numeric= 0.7854
# 출력: l=3: closed= 0.0000 (     0)  numeric=-0.0000
# 출력: l=4: closed=-0.1309 (-pi/24)  numeric=-0.1309
# 출력: l=5: closed= 0.0000 (     0)  numeric=-0.0000
# 출력: l=6: closed= 0.0491 ( pi/64)  numeric= 0.0491

# %% [markdown]
# ## 4. 왜 홀수 차수($l\ge3$)는 0인가?
#
# $x=\cos\theta$ 로 치환하면 $A_l\propto\int_0^1 x\,P_l(x)\,dx$ 이다.
# $x P_l(x)$ 는 $l$ 이 홀수이면 **짝함수** 이므로 $\int_0^1 = \tfrac12\int_{-1}^1 x P_l(x)dx$ 인데,
# $x=P_1(x)$ 와 $P_l$ ($l\ne1$)은 $[-1,1]$ 에서 직교하므로 0이 된다.
# $l=1$ 만 $\int_{-1}^1 P_1^2\,dx=2/3\neq0$ 이라 예외적으로 살아남는다.
# 짝수 $l$ 에서는 $xP_l$ 이 홀함수라 이 논리가 적용되지 않아 0이 아니다.

# %%
x = np.linspace(-1, 1, 200001)
for l in range(1, 7):
    P = np.polynomial.legendre.Legendre.basis(l)
    half = np.trapezoid(x[x >= 0] * P(x[x >= 0]), x[x >= 0])
    full = np.trapezoid(x * P(x), x)
    print(f"l={l}: int_0^1 x P_l = {half: .4f},  int_-1^1 x P_l = {full: .4f}")
# 출력: l=1: int_0^1 x P_l =  0.3333,  int_-1^1 x P_l =  0.6667
# 출력: l=2: int_0^1 x P_l =  0.1250,  int_-1^1 x P_l = -0.0000
# 출력: l=3: int_0^1 x P_l =  0.0000,  int_-1^1 x P_l =  0.0000
# 출력: l=4: int_0^1 x P_l = -0.0208,  int_-1^1 x P_l = -0.0000
# 출력: l=5: int_0^1 x P_l =  0.0000,  int_-1^1 x P_l =  0.0000
# 출력: l=6: int_0^1 x P_l =  0.0078,  int_-1^1 x P_l = -0.0000

# %% [markdown]
# ## 5. 시각화 — 급격히 감소하는 $\hat{A_l}$
#
# $l\le2$ 까지의 세 값이 에너지를 거의 다 차지하므로, Irradiance 는 9개 계수(RGB 27개)로 충분히 근사된다.
# 오른쪽은 $\max(\cos\theta,0)$ 을 $l\le2$ 까지의 SH 로 재구성한 결과다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("$\\hat{A_l}$ vs l", "max(cos θ, 0) 와 l≤2 SH 재구성"))
ls = list(range(7))
fig.add_trace(go.Bar(x=ls, y=A_hat_vals, text=[f"{v:.4f}" for v in A_hat_vals],
                     textposition="outside", name="Â_l"), row=1, col=1)

th = np.linspace(0, math.pi, 400)
recon = np.zeros_like(th)
for l in range(3):
    # 클램프 코사인 자체의 SH 계수는 A_l ; 재구성 = Σ A_l Y_l^0
    A_l = A_hat_vals[l] / math.sqrt(4 * math.pi / (2 * l + 1))
    recon += A_l * Y_l0(l, th)
fig.add_trace(go.Scatter(x=np.degrees(th), y=np.maximum(np.cos(th), 0), name="max(cos θ,0)"), row=1, col=2)
fig.add_trace(go.Scatter(x=np.degrees(th), y=recon, name="l≤2 재구성", line=dict(dash="dash")), row=1, col=2)
fig.update_xaxes(title_text="l", row=1, col=1)
fig.update_xaxes(title_text="θ (deg)", row=1, col=2)
fig.update_layout(width=1000, height=420, title="클램프 코사인의 SH 계수 Â_l 와 저차수 근사")
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"))
print("saved expy.png; max recon error =", float(np.abs(recon - np.maximum(np.cos(th), 0)).max()).__round__(4))
# 출력: saved expy.png; max recon error = 0.0918
