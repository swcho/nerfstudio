# %% [markdown]
# # `P(int l, int m, double x)` — 세 재귀식의 적용 순서
#
# 버금 르장드르 다항식 $P_l^m(x)$를 계산하는 코드는 세 재귀식을 **정해진 순서**로 사용한다.
#
# 1. **시작점** (이전 결과 불필요) : $P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}$ → 변수 `pmm`
# 2. **한 단계 올리기** : $P_{m+1}^m = x(2m+1) P_m^m$ → 변수 `pmmp1`
# 3. **일반 재귀** : $(l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m$ → 변수 `pll`, `ll = m+2 .. l` 반복
#
# 각 단계 후에 `l == m`, `l == m+1` 이면 **조기 반환**한다. 아래에서 단계별로 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido, scipy(검증용, 없으면 건너뜀)
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
# ## 1단계: `pmm` — $P_m^m$ 을 반복문으로 계산
#
# `fact`는 1, 3, 5, … 로 증가하며 이중 계승 $(2m-1)!!$ 을 만들고, `-fact`의 부호가 $(-1)^m$ 을, `somx2 = \sqrt{1-x^2}` 가 $m$번 곱해져 $(1-x^2)^{m/2}$ 을 만든다.

# %%
def step1_pmm(m, x, trace=None):
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            if trace is not None:
                trace.append(f"  i={i}: fact={fact:.0f}, pmm={pmm:+.6f}")
            fact += 2.0
    return pmm


x = 0.5
trace = []
pmm = step1_pmm(3, x, trace)
print("\n".join(trace))
closed = (-1) ** 3 * (5 * 3 * 1) * (1 - x**2) ** 1.5
print(f"P_3^3(0.5) 코드={pmm:.6f}, 닫힌식={closed:.6f}")
# 출력:
#   i=1: fact=1, pmm=-0.866025
#   i=2: fact=3, pmm=+2.250000
#   i=3: fact=5, pmm=-9.742786
# P_3^3(0.5) 코드=-9.742786, 닫힌식=-9.742786

# %% [markdown]
# ## 2단계: `pmmp1` — $P_{m+1}^m = x(2m+1)P_m^m$
#
# 곱셈 한 번으로 band를 하나 올린다. `l == m+1` 이면 여기서 반환.

# %%
def step2_pmmp1(m, x, pmm):
    return x * (2.0 * m + 1.0) * pmm


m = 1
pmm = step1_pmm(m, x)
pmmp1 = step2_pmmp1(m, x, pmm)
print(f"P_1^1(0.5) = {pmm:.6f}")
print(f"P_2^1(0.5) = x*(2m+1)*P_1^1 = {x}*{2*m+1}*{pmm:.6f} = {pmmp1:.6f}")
# 출력:
# P_1^1(0.5) = -0.866025
# P_2^1(0.5) = x*(2m+1)*P_1^1 = 0.5*3*-0.866025 = -1.299038

# %% [markdown]
# ## 3단계: `pll` — 일반 재귀를 `ll = m+2` 부터 `l` 까지 반복
#
# $$P_{ll}^m = \frac{(2\,ll-1)\,x\,P_{ll-1}^m - (ll+m-1)\,P_{ll-2}^m}{ll-m}$$
#
# 매 반복 뒤 `pmm ← pmmp1`, `pmmp1 ← pll` 로 두 칸 창(window)을 밀어 올린다.

# %%
def P(l, m, x, trace=None):
    """원문 C 코드의 1:1 파이썬 이식 (trace 리스트에 단계 기록)."""
    pmm = step1_pmm(m, x)
    if trace is not None:
        trace.append(f"[1] pmm   = P_{m}^{m} = {pmm:+.6f}")
    if l == m:
        if trace is not None:
            trace.append("    l == m → pmm 반환")
        return pmm
    pmmp1 = step2_pmmp1(m, x, pmm)
    if trace is not None:
        trace.append(f"[2] pmmp1 = P_{m+1}^{m} = {pmmp1:+.6f}")
    if l == m + 1:
        if trace is not None:
            trace.append("    l == m+1 → pmmp1 반환")
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        if trace is not None:
            trace.append(f"[3] ll={ll}: pll = P_{ll}^{m} = {pll:+.6f}   (pmm←pmmp1, pmmp1←pll)")
        pmm = pmmp1
        pmmp1 = pll
    if trace is not None:
        trace.append("    반복 종료 → pll 반환")
    return pll


for (l, m) in [(2, 2), (3, 2), (5, 1)]:
    tr = []
    val = P(l, m, 0.5, tr)
    print(f"--- P({l},{m},0.5) = {val:.6f}")
    print("\n".join(tr))
# 출력:
# --- P(2,2,0.5) = 2.250000
# [1] pmm   = P_2^2 = +2.250000
#     l == m → pmm 반환
# --- P(3,2,0.5) = 5.625000
# [1] pmm   = P_2^2 = +2.250000
# [2] pmmp1 = P_3^2 = +5.625000
#     l == m+1 → pmmp1 반환
# --- P(5,1,0.5) = 1.928260
# [1] pmm   = P_1^1 = -0.866025
# [2] pmmp1 = P_2^1 = -1.299038
# [3] ll=3: pll = P_3^1 = -0.324760   (pmm←pmmp1, pmmp1←pll)
# [3] ll=4: pll = P_4^1 = +1.353165   (pmm←pmmp1, pmmp1←pll)
# [3] ll=5: pll = P_5^1 = +1.928260   (pmm←pmmp1, pmmp1←pll)
#     반복 종료 → pll 반환

# %% [markdown]
# ## 검증: scipy의 `lpmv` 와 비교
#
# 원문 코드는 Condon–Shortley 위상 $(-1)^m$ 을 포함하며 scipy `lpmv`도 동일 규약이다.

# %%
try:
    from scipy.special import lpmv
    max_err = 0.0
    for l in range(0, 7):
        for m in range(0, l + 1):
            for xx in np.linspace(-0.99, 0.99, 9):
                max_err = max(max_err, abs(P(l, m, xx) - lpmv(m, l, xx)))
    print(f"l<=6 전체에서 scipy.lpmv 대비 최대 오차 = {max_err:.2e}")
except ImportError:
    print("scipy 없음 — 검증 건너뜀")
# 출력:
# l<=6 전체에서 scipy.lpmv 대비 최대 오차 = 2.73e-12

# %% [markdown]
# ## 시각화: 어떤 (l, m) 이 어느 단계에서 반환되는가
#
# - 대각선 $l=m$ : 1단계(`pmm`)에서 반환
# - 바로 아래 $l=m+1$ : 2단계(`pmmp1`)에서 반환
# - 그 외 : 3단계 반복이 $l-m-1$ 번 돈 뒤 `pll` 반환
#
# 오른쪽에는 고정한 $m=1$ 에 대해 $P_l^1(x)$ 곡선이 band $l$ 이 올라갈수록 어떻게 쌓이는지 그린다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

L = 6
z = np.full((L + 1, L + 1), np.nan)
txt = [["" for _ in range(L + 1)] for _ in range(L + 1)]
for l in range(L + 1):
    for m in range(l + 1):
        iters = max(0, l - m - 1)
        z[l, m] = min(iters, 2) if l != m else 0  # 0: pmm, 1: pmmp1, >=2: pll(3단계)
        if l == m:
            z[l, m] = 0; txt[l][m] = "pmm"
        elif l == m + 1:
            z[l, m] = 1; txt[l][m] = "pmmp1"
        else:
            z[l, m] = 2; txt[l][m] = f"pll×{iters}"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("반환 단계 (행 l, 열 m)", "P_l^1(x), l=1..6"),
    column_widths=[0.45, 0.55],
)
fig.add_trace(go.Heatmap(
    z=z, text=txt, texttemplate="%{text}",
    x=[f"m={m}" for m in range(L + 1)], y=[f"l={l}" for l in range(L + 1)],
    colorscale=[[0, "#4c78a8"], [0.5, "#f58518"], [1, "#54a24b"]],
    zmin=0, zmax=2, showscale=False,
), row=1, col=1)
fig.update_yaxes(autorange="reversed", row=1, col=1)

xs = np.linspace(-1, 1, 201)
for l in range(1, L + 1):
    fig.add_trace(go.Scatter(x=xs, y=[P(l, 1, v) for v in xs], mode="lines",
                             name=f"P_{l}^1"), row=1, col=2)
fig.update_xaxes(title_text="x", row=1, col=2)
fig.update_layout(title="P(l,m,x): 1단계 pmm → 2단계 pmmp1 → 3단계 pll 반복", width=1100, height=500)
_show(fig)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out, scale=2)
print("saved", out)
# 출력: saved /home/sungwoo/projects/swcho/nerfstudio/fm/.fm/hints/5bcf1724-c2b9-4068-a765-841058eca029/expy.png
