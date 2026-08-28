# %% [markdown]
# # 이중 계승 $n!!$ 계산하기
#
# 이중 계승(double factorial)은 $n$에서 시작해 **2씩 줄여가며** 1 또는 2까지 곱하는 연산이다.
#
# $$ n!! = \begin{cases} n \cdot (n-2) \cdots 5 \cdot 3 \cdot 1, & n>0 \text{ odd} \\ n \cdot (n-2) \cdots 6 \cdot 4 \cdot 2, & n>0 \text{ even} \end{cases} $$
#
# 구면 조화 함수(spherical harmonics)의 버금 르장드르 함수 시작점
# $P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}$ 에 등장한다.
#
# 필요 패키지: numpy, plotly, kaleido, scipy(선택)

# %%
import os
import math
import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 정의 그대로 구현 — 2씩 줄여가며 곱한다

# %%
def double_factorial(n: int) -> int:
    """n!! : n, n-2, n-4, ... (1 또는 2까지) 의 곱. 관례상 0!! = (-1)!! = 1."""
    if n <= 0:
        return 1
    result = 1
    k = n
    while k > 0:
        result *= k
        k -= 2
    return result


for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    terms = list(range(n, 0, -2))
    print(f"{n:2d}!! = {' * '.join(map(str, terms)):>22s} = {double_factorial(n)}")
# 출력:
#  1!! =                      1 = 1
#  2!! =                      2 = 2
#  3!! =                  3 * 1 = 3
#  4!! =                  4 * 2 = 8
#  5!! =              5 * 3 * 1 = 15
#  6!! =              6 * 4 * 2 = 48
#  7!! =          7 * 5 * 3 * 1 = 105
#  8!! =          8 * 6 * 4 * 2 = 384
#  9!! =      9 * 7 * 5 * 3 * 1 = 945
# 10!! =     10 * 8 * 6 * 4 * 2 = 3840

# %% [markdown]
# ## 2. 재귀 정의와 일반 계승과의 관계
#
# 재귀형: $n!! = n \cdot (n-2)!!$, $0!! = 1!! = 1$.
#
# 일반 계승과의 관계:
# - 짝수 $n=2k$: $(2k)!! = 2^k \, k!$
# - 홀수 $n=2k-1$: $(2k-1)!! = \dfrac{(2k)!}{2^k \, k!}$
#
# 그리고 $n!! \cdot (n-1)!! = n!$ 이 항상 성립한다.

# %%
def double_factorial_rec(n: int) -> int:
    return 1 if n <= 0 else n * double_factorial_rec(n - 2)


ok = True
for n in range(1, 16):
    a = double_factorial(n)
    b = double_factorial_rec(n)
    if n % 2 == 0:
        k = n // 2
        c = 2**k * math.factorial(k)
    else:
        k = (n + 1) // 2
        c = math.factorial(2 * k) // (2**k * math.factorial(k))
    d = a * double_factorial(n - 1)  # n!! * (n-1)!! == n!
    ok &= (a == b == c) and (d == math.factorial(n))
print("반복 == 재귀 == 계승 공식, 그리고 n!!*(n-1)!! == n! :", ok)
# 출력: 반복 == 재귀 == 계승 공식, 그리고 n!!*(n-1)!! == n! : True

# %% [markdown]
# ## 3. scipy 와 비교 (설치되어 있으면)

# %%
try:
    from scipy.special import factorial2
    vals = [int(factorial2(n, exact=True)) for n in range(1, 11)]
    mine = [double_factorial(n) for n in range(1, 11)]
    print("scipy factorial2 :", vals)
    print("내 구현           :", mine)
    print("일치:", vals == mine)
except ImportError:
    print("scipy 없음 — 건너뜀")
# 출력:
# scipy factorial2 : [1, 2, 3, 8, 15, 48, 105, 384, 945, 3840]
# 내 구현           : [1, 2, 3, 8, 15, 48, 105, 384, 945, 3840]
# 일치: True

# %% [markdown]
# ## 4. 구면 조화 함수에서의 사용: $P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}$
#
# 원문에서 이중 계승은 버금 르장드르 함수의 시작점(대각 항)에 등장한다.
# $(2m-1)!!$ 은 항상 홀수 이중 계승이다.

# %%
def P_mm(m: int, x):
    return (-1) ** m * double_factorial(2 * m - 1) * (1 - x**2) ** (m / 2)


for m in range(0, 5):
    print(f"m={m}: (2m-1)!! = ({2*m-1})!! = {double_factorial(2*m-1):4d},  P_{m}^{m}(0) = {P_mm(m, 0.0):.0f}")
# 출력:
# m=0: (2m-1)!! = (-1)!! =    1,  P_0^0(0) = 1
# m=1: (2m-1)!! = (1)!! =    1,  P_1^1(0) = -1
# m=2: (2m-1)!! = (3)!! =    3,  P_2^2(0) = 3
# m=3: (2m-1)!! = (5)!! =   15,  P_3^3(0) = -15
# m=4: (2m-1)!! = (7)!! =  105,  P_4^4(0) = 105

# %% [markdown]
# ## 5. 시각화 — $n!!$ 와 $n!$ 의 성장 비교 (로그 스케일)
#
# 홀수/짝수 두 갈래로 나뉘어 각각 $n!$ 보다 훨씬 느리게(대략 $\sqrt{n!}$ 규모로) 커진다.

# %%
import plotly.graph_objects as go

ns = np.arange(1, 21)
df_vals = np.array([double_factorial(int(n)) for n in ns], dtype=float)
f_vals = np.array([math.factorial(int(n)) for n in ns], dtype=float)

fig = go.Figure()
fig.add_trace(go.Scatter(x=ns, y=f_vals, mode="lines+markers", name="n!",
                         line=dict(color="gray", dash="dash")))
fig.add_trace(go.Scatter(x=ns[ns % 2 == 1], y=df_vals[ns % 2 == 1], mode="lines+markers",
                         name="n!! (홀수: ...·5·3·1)", line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=ns[ns % 2 == 0], y=df_vals[ns % 2 == 0], mode="lines+markers",
                         name="n!! (짝수: ...·6·4·2)", line=dict(color="royalblue")))
fig.update_layout(title="이중 계승 n!! 과 계승 n! 의 성장 비교",
                  xaxis_title="n", yaxis_title="값 (log)", yaxis_type="log",
                  width=800, height=500, template="plotly_white")

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out_png)
print("저장:", out_png)
_show(fig)
# 출력: 저장: .../273db38d-8573-4bfd-ba3c-97a32d588a27/expy.png
