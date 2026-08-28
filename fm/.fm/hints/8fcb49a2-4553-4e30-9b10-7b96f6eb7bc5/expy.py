# %% [markdown]
# # 투영(Projection): 기저 함수의 계수를 적분으로 구하기
#
# 원본 함수 $f(x)$를 기저 함수 $b_i(x)$의 선형 조합으로 근사한다.
#
# $$f(x)\approx\sum_i c_i b_i(x),\qquad c_i=\int f(x)\,b_i(x)\,dx$$
#
# 이 스크립트는 (1) 벡터 내적과의 유사성, (2) 직교성이 왜 필요한지,
# (3) 푸리에(sin/cos) 기저로 실제 투영·복원을 단계적으로 보여준다.

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
# ## 1. 벡터 버전: 성분 = 내적
#
# $\vec v=(3,4)$의 성분은 단위축과의 내적으로 나온다: $c_i=\vec v\cdot\hat e_i$.

# %%
v = np.array([3.0, 4.0])
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
c1, c2 = v @ e1, v @ e2
print("c1 =", c1, ", c2 =", c2)
print("복원:", c1 * e1 + c2 * e2)
# 출력: c1 = 3.0 , c2 = 4.0
# 출력: 복원: [3. 4.]

# %% [markdown]
# ## 2. 함수 버전: 내적 = 적분
#
# 구간 $[-\pi,\pi]$에서 정규화된 푸리에 기저
# $b_0=\frac{1}{\sqrt{2\pi}},\ b_{2k-1}=\frac{\sin kx}{\sqrt\pi},\ b_{2k}=\frac{\cos kx}{\sqrt\pi}$
# 를 만들고, 직교성 $\int b_ib_j\,dx=\delta_{ij}$를 수치 적분으로 확인한다.

# %%
x = np.linspace(-np.pi, np.pi, 4001)
dx = x[1] - x[0]


def integrate(y):
    """구간 전체에 걸친 정적분 (사다리꼴 공식)."""
    return np.trapezoid(y, dx=dx)


def basis(n_terms):
    B = [np.full_like(x, 1 / np.sqrt(2 * np.pi))]
    k = 1
    while len(B) < n_terms:
        B.append(np.sin(k * x) / np.sqrt(np.pi))
        if len(B) < n_terms:
            B.append(np.cos(k * x) / np.sqrt(np.pi))
        k += 1
    return np.array(B)


B = basis(5)
gram = np.array([[integrate(bi * bj) for bj in B] for bi in B])
print(np.round(gram, 6))
# 출력: 5x5 단위행렬 (대각 1, 나머지 0)
# 출력: [[ 1.  0. -0. -0.  0.]
# 출력:  [ 0.  1.  0.  0.  0.]
# 출력:  [-0.  0.  1.  0. -0.]
# 출력:  [-0.  0.  0.  1.  0.]
# 출력:  [ 0.  0. -0.  0.  1.]]

# %% [markdown]
# ## 3. 투영: $c_i=\int f(x)b_i(x)\,dx$
#
# 원본 함수로 톱니 모양의 $f(x)=x$ 를 쓴다(불연속이라 고차항이 오래 남아 근사 과정이 잘 보인다).
# 각 계수는 다른 계수와 무관하게 **적분 한 번**으로 구해진다.

# %%
f = x.copy()  # f(x) = x

N = 21
B = basis(N)
coeffs = np.array([integrate(f * b) for b in B])  # <- 투영
for i, c in enumerate(coeffs[:7]):
    print(f"c_{i} = {c:+.4f}")
# 출력: c_0 = +0.0000   (상수항: f가 기함수여서 0)
# 출력: c_1 = +3.5449   (sin x)   = 2*sqrt(pi)
# 출력: c_2 = -0.0000   (cos x)
# 출력: c_3 = -1.7725   (sin 2x)  = -sqrt(pi)
# 출력: c_4 = +0.0000
# 출력: c_5 = +1.1816   (sin 3x)  = 2*sqrt(pi)/3
# 출력: c_6 = -0.0000

# 해석적 값과 비교: f(x)=x 의 sin(kx) 계수는 (-1)^{k+1} 2 sqrt(pi) / k
k = 1
analytic = (-1) ** (k + 1) * 2 * np.sqrt(np.pi) / k
print("해석값 c_1 =", round(analytic, 4))
# 출력: 해석값 c_1 = 3.5449

# %% [markdown]
# ## 4. 복원: $f(x)\approx\sum_{i<n} c_i b_i(x)$
#
# 항의 수 $n$을 늘리면 근사 오차가 줄어든다.

# %%
def reconstruct(n):
    return coeffs[:n] @ B[:n]


for n in [3, 5, 11, 21]:
    err = np.sqrt(integrate((f - reconstruct(n)) ** 2))
    print(f"n={n:2d}  L2 오차 = {err:.4f}")
# 출력: n= 3  L2 오차 = 2.8468
# 출력: n= 5  L2 오차 = 2.2278
# 출력: n=11  L2 오차 = 1.5095
# 출력: n=21  L2 오차 = 1.0936

# %% [markdown]
# ## 5. 시각화
#
# 왼쪽: 원문 그림처럼 $f\times b_i$ 의 곱(음영)을 적분한 값이 $c_i$.
# 오른쪽: 항의 수를 늘려 가며 복원한 결과.

# %%
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {"rowspan": 3}], [{}, None], [{}, None]],
    subplot_titles=[
        f"∫ f·b₁ dx = c₁ = {coeffs[1]:+.2f}", "복원  Σ cᵢ bᵢ(x)",
        f"∫ f·b₃ dx = c₃ = {coeffs[3]:+.2f}",
        f"∫ f·b₅ dx = c₅ = {coeffs[5]:+.2f}",
    ],
    horizontal_spacing=0.08, vertical_spacing=0.12,
)
for row, i in enumerate([1, 3, 5], start=1):
    fig.add_trace(go.Scatter(x=x, y=f, line=dict(color="#888", width=1.5),
                             name="f(x)=x", showlegend=(row == 1)), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=B[i], line=dict(color="#1f77b4", width=1.5, dash="dot"),
                             name="bᵢ(x)", showlegend=(row == 1)), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=f * B[i], fill="tozeroy",
                             line=dict(color="#d62728", width=1),
                             name="f·bᵢ", showlegend=(row == 1)), row=row, col=1)

fig.add_trace(go.Scatter(x=x, y=f, line=dict(color="black", width=2), name="원본 f"), row=1, col=2)
for n, col in [(3, "#9ecae1"), (5, "#6baed6"), (11, "#3182bd"), (21, "#08519c")]:
    fig.add_trace(go.Scatter(x=x, y=reconstruct(n), line=dict(color=col, width=1.5),
                             name=f"n={n}"), row=1, col=2)

fig.update_layout(title="투영(적분으로 계수 구하기) → 복원", width=1100, height=650,
                  template="plotly_white")
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"))
print("saved expy.png")
# 출력: saved expy.png
