# %% [markdown]
# # 버금 르장드르 다항식의 재귀식 실험
#
# 세 가지 식으로 $P_l^m(x)$ 를 계산합니다.
#
# 1. 시작점: $P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}$
# 2. 한 단계 위: $P_{m+1}^m = x(2m+1)P_m^m$
# 3. 일반 $l$: $(l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m$
#
# 필요 패키지: numpy, plotly, kaleido, scipy(검증용, 없어도 됨)

# %%
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
# ## 1단계: 시작점 $P_m^m$ (식 1)
#
# 이중 계승 $(2m-1)!!$ 을 누적곱으로 계산합니다. 원문 C 코드의 `pmm` 루프와 같습니다.


# %%
def P_mm(m, x):
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    somx2 = np.sqrt((1.0 - x) * (1.0 + x))
    fact = 1.0
    for _ in range(m):
        pmm = pmm * (-fact) * somx2
        fact += 2.0
    return pmm


x0 = 0.5
for m in range(4):
    print(f"P_{m}^{m}({x0}) = {P_mm(m, x0):.6f}")
# 출력:
# P_0^0(0.5) = 1.000000
# P_1^1(0.5) = -0.866025
# P_2^2(0.5) = 2.250000
# P_3^3(0.5) = -9.742786

# %% [markdown]
# ## 2단계: 한 칸 아래 $P_{m+1}^m$ (식 2)
#
# 곱셈 한 번: $P_{m+1}^m = x(2m+1)P_m^m$.  $m=0$ 이면 $P_1^0 = x$.


# %%
def P_mp1_m(m, x):
    return np.asarray(x, dtype=float) * (2.0 * m + 1.0) * P_mm(m, x)


for m in range(3):
    print(f"P_{m+1}^{m}({x0}) = {P_mp1_m(m, x0):.6f}")
# 출력:
# P_1^0(0.5) = 0.500000
# P_2^1(0.5) = -1.299038
# P_3^2(0.5) = 5.625000

# %% [markdown]
# ## 3단계: 같은 열에서 계속 내려가기 (식 3)
#
# $$P_l^m = \frac{x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m}{l-m}$$
#
# 두 개의 직전 항(`pmm`, `pmmp1`)만 기억하면 되므로 피보나치 수열처럼 반복문으로 처리합니다.


# %%
def P(l, m, x):
    """원문 C 코드 P(l,m,x)의 파이썬 버전 (식 1 -> 2 -> 3)."""
    x = np.asarray(x, dtype=float)
    pmm = P_mm(m, x)
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = None
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


print("m=0 열 (보통의 르장드르 다항식):")
for l in range(5):
    print(f"  P_{l}^0(0.5) = {P(l, 0, x0):.6f}")
print("닫힌형 비교: P_2 =", (3 * x0**2 - 1) / 2, " P_3 =", (5 * x0**3 - 3 * x0) / 2)
# 출력:
# m=0 열 (보통의 르장드르 다항식):
#   P_0^0(0.5) = 1.000000
#   P_1^0(0.5) = 0.500000
#   P_2^0(0.5) = -0.125000
#   P_3^0(0.5) = -0.437500
#   P_4^0(0.5) = -0.289062
# 닫힌형 비교: P_2 = -0.125  P_3 = -0.4375

# %% [markdown]
# ## 4단계: scipy 구현과 비교 (있을 때만)
#
# `scipy.special.lpmv(m, l, x)` 는 Condon-Shortley 위상 $(-1)^m$ 을 포함하므로 위 재귀식과 그대로 일치해야 합니다.

# %%
try:
    from scipy.special import lpmv
    max_err = 0.0
    xs = np.linspace(-0.99, 0.99, 50)
    for l in range(6):
        for m in range(l + 1):
            max_err = max(max_err, np.max(np.abs(P(l, m, xs) - lpmv(m, l, xs))))
    print(f"l<=5 전체, scipy와 최대 오차 = {max_err:.2e}")
except ImportError:
    print("scipy 없음 - 비교 생략")
# 출력:
# l<=5 전체, scipy와 최대 오차 = 3.41e-13

# %% [markdown]
# ## 5단계: 시각화
#
# 왼쪽: $m=0$ 열을 식 3으로 내려가며 만든 $P_l^0$.  오른쪽: $m=1$ 열 ($P_1^1$ 은 식 1, $P_2^1$ 은 식 2, 그 뒤는 식 3).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

xs = np.linspace(-1, 1, 400)
fig = make_subplots(rows=1, cols=2, subplot_titles=("m = 0 열", "m = 1 열"))
for l in range(0, 5):
    fig.add_trace(go.Scatter(x=xs, y=P(l, 0, xs), name=f"P_{l}^0"), row=1, col=1)
for l in range(1, 5):
    src = "식1" if l == 1 else ("식2" if l == 2 else "식3")
    fig.add_trace(go.Scatter(x=xs, y=P(l, 1, xs), name=f"P_{l}^1 ({src})"), row=1, col=2)
fig.update_layout(title="재귀식으로 만든 버금 르장드르 다항식", width=1000, height=450)
fig.update_xaxes(title_text="x")
_show(fig)
out = os.path.join(HERE, "expy.png")
fig.write_image(out)
print("saved", out)
# 출력: saved .../expy.png
