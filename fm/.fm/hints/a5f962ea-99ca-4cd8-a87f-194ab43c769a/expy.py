# %% [markdown]
# # D7 배경 합성 — 왜 배경색을 매 스텝 랜덤으로 바꾸는가?
#
# splatfacto의 `get_outputs()` 마지막 단계(D7)는 래스터라이저가 돌려준 `render`(premultiplied 색)와
# `alpha`(누적 불투명도)에 배경색 $b$를 합성합니다:
#
# $$
# \text{rgb}(p) = \operatorname{clamp}\big(\text{render}(p) + (1-\text{alpha}(p))\,b,\ 0,\ 1\big),
# \qquad b \sim U[0,1]^3 \ (\text{training, background\_color="random"})
# $$
#
# 이 노트북은 "픽셀 하나 = 가우시안 하나"로 극단적으로 단순화한 1D 토이로,
# **고정 배경**에서는 두 해(빈 공간 vs 배경색 가우시안)가 loss 상 구분되지 않고,
# **랜덤 배경**에서는 기대 loss가 올바른 해(alpha=0 또는 alpha=1)를 유일하게 선호함을 수치로 확인합니다.

# %%
# 필요 패키지: numpy, plotly, kaleido (PNG 저장용; `python -c "import kaleido; kaleido.get_chrome_sync()"` 로 크롬 설치 필요할 수 있음)
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 합성 공식 그대로 구현
#
# 가우시안 하나가 픽셀을 덮는 경우 `render = alpha * c` (premultiplied), `alpha = a` 입니다.
# nerfstudio 코드(`splatfacto.py`)와 동일하게 `rgb = clamp(render + (1-alpha)*b, 0, 1)` 로 합성합니다.
# GT가 RGBA(알파 채널 있음)면 `composite_with_background()`가 **같은 b**로 GT도 합성합니다.

# %%
def composite(render, alpha, b):
    """D7: rgb = clamp(render + (1-alpha) * b, 0, 1)"""
    return np.clip(render + (1.0 - alpha) * b, 0.0, 1.0)


def composite_gt(gt_rgb, gt_alpha, b):
    """composite_with_background(): RGBA GT일 때 alpha*rgb + (1-alpha)*b"""
    return gt_alpha * gt_rgb + (1.0 - gt_alpha) * b


def mse(x, y):
    return float(np.mean((x - y) ** 2))


b = np.array([0.2, 0.7, 0.4])
a, c = 0.6, np.array([1.0, 0.0, 0.0])          # 60% 불투명한 빨간 가우시안
print("render(premult) =", a * c)
print("rgb             =", composite(a * c, a, b))   # 0.6*빨강 + 0.4*배경
# 출력: render(premult) = [0.6 0.  0. ]
# 출력: rgb             = [0.68 0.28 0.16]

# %% [markdown]
# ## 2. 케이스 A — GT가 "검은 배경 위의 물체"(RGBA, 빈 공간 존재)
#
# 빈 공간 픽셀의 GT는 `gt_alpha = 0` 입니다. 후보 해 두 가지:
#
# - **해 ①(정답)**: 가우시안 없음, $a=0$ → $\text{rgb} = b$
# - **해 ②(가짜)**: 검은 가우시안으로 덮음, $a=1,\ c=0$ → $\text{rgb} = 0$
#
# 고정 검은 배경($b=0$)이면 GT도 pred도 모두 0이라 두 해의 loss가 **완전히 같습니다**(중간 $a$도 전부 0).
# 랜덤 배경이면 GT는 $b$가 되고 pred는 $(1-a)\,b$ 이므로
#
# $$
# \mathbb{E}_b\big[\|(1-a)b - b\|^2\big] = a^2\,\mathbb{E}[b^2] = \tfrac{a^2}{3}\ (\text{채널당}),
# $$
#
# $a=0$ 이 유일한 최소가 됩니다. 즉 "배경색 흉내"는 배경이 바뀌는 순간 들통납니다.

# %%
gt_rgb_empty, gt_alpha_empty = np.zeros(3), 0.0    # 빈 공간
c_black = np.zeros(3)
alphas = np.linspace(0, 1, 11)

# (1) 고정 검은 배경
b_black = np.zeros(3)
loss_fixed_A = [mse(composite(a * c_black, a, b_black),
                    composite_gt(gt_rgb_empty, gt_alpha_empty, b_black)) for a in alphas]
print("고정 검은 배경 | alpha별 loss:", np.round(loss_fixed_A, 4))

# (2) 랜덤 배경 — 몬테카를로 기대 loss
B = rng.random((20000, 3))
def expected_loss(a, c, gt_rgb, gt_alpha, B):
    pred = composite(a * c, a, B)
    gt = composite_gt(gt_rgb, gt_alpha, B)
    return float(np.mean((pred - gt) ** 2))
loss_rand_A = [expected_loss(a, c_black, gt_rgb_empty, gt_alpha_empty, B) for a in alphas]
print("랜덤 배경     | alpha별 E[loss]:", np.round(loss_rand_A, 4))
print("이론값 a^2/3            :", np.round(alphas ** 2 / 3, 4))
# 출력: 고정 검은 배경 | alpha별 loss: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 출력: 랜덤 배경     | alpha별 E[loss]: [0.     0.0033 0.0134 0.03   0.0534 0.0835 0.1202 0.1636 0.2137 0.2704 0.3339]
# 출력: 이론값 a^2/3            : [0.     0.0033 0.0133 0.03   0.0533 0.0833 0.12   0.1633 0.2133 0.27   0.3333]

# %% [markdown]
# ## 3. 케이스 B — GT가 일반 사진(RGB, 알파 없음), 벽 색이 배경색과 같을 때
#
# 실제 학습 데이터는 대부분 RGB 사진이라 "빈 공간"이 없습니다. 이때 위험은 반대 방향입니다:
# 벽 색 $g$가 고정 배경색과 같으면 가우시안을 **아예 두지 않아도**($a=0$) 배경이 색을 대신 채워
# loss가 0 → 지오메트리에 구멍이 생깁니다(다른 시점/다른 배경으로 렌더하면 뚫려 보임).
#
# 랜덤 배경이면 pred $=(1-a)b + a\,g$, GT $=g$ 이므로
#
# $$
# \mathbb{E}_b\big[\|(1-a)(b-g)\|^2\big] = (1-a)^2\,\mathbb{E}[(b-g)^2],
# $$
#
# $a=1$ 만이 loss 0 → 내용이 있는 곳은 반드시 덮도록 강제됩니다.

# %%
g = np.array([0.5, 0.5, 0.5])                       # 회색 벽
gt_rgb_wall, gt_alpha_wall = g, 1.0                 # RGB GT: alpha=1 로 취급(composite_with_background는 no-op)

loss_fixed_B = [mse(composite(a * g, a, g), gt_rgb_wall) for a in alphas]   # 고정 배경색 == 벽 색
loss_rand_B = [expected_loss(a, g, gt_rgb_wall, gt_alpha_wall, B) for a in alphas]
print("고정 배경(=벽색) | alpha별 loss:", np.round(loss_fixed_B, 4))
print("랜덤 배경        | alpha별 E[loss]:", np.round(loss_rand_B, 4))
print("이론값 (1-a)^2*E[(b-g)^2]  :", np.round((1 - alphas) ** 2 * np.mean((B - g) ** 2), 4))
# 출력: 고정 배경(=벽색) | alpha별 loss: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 출력: 랜덤 배경        | alpha별 E[loss]: [0.0831 0.0673 0.0532 0.0407 0.0299 0.0208 0.0133 0.0075 0.0033 0.0008 0.    ]
# 출력: 이론값 (1-a)^2*E[(b-g)^2]  : [0.0831 0.0673 0.0532 0.0407 0.0299 0.0208 0.0133 0.0075 0.0033 0.0008 0.    ]

# %% [markdown]
# ## 4. 경사하강 시뮬레이션 — 실제로 alpha가 어디로 가는가
#
# $a = \sigma(\theta)$ 로 두고 매 스텝 배경 $b$ 를 새로 뽑아(실제 학습처럼) SGD 합니다.
# - 케이스 A(빈 공간, 검은 가우시안 $a_0=0.9$): 고정 배경이면 기울기 0 → 그대로 남음, 랜덤이면 0으로 수렴
# - 케이스 B(회색 벽, $a_0=0.1$): 고정 배경(=벽색)이면 구멍 유지, 랜덤이면 1로 수렴

# %%
def sgd_alpha(theta0, c, gt_rgb, gt_alpha, random_bg, fixed_b, steps=400, lr=2.0):
    theta, hist = theta0, []
    for _ in range(steps):
        bb = rng.random(3) if random_bg else fixed_b
        a = 1 / (1 + np.exp(-theta))
        pred = a * c + (1 - a) * bb                    # clamp 생략(범위 내)
        gt = gt_alpha * gt_rgb + (1 - gt_alpha) * bb
        dL_da = np.mean(2 * (pred - gt) * (c - bb))    # d/da of mean((pred-gt)^2)
        theta -= lr * dL_da * a * (1 - a)
        hist.append(a)
    return np.array(hist)

logit = lambda p: np.log(p / (1 - p))
trajA_fixed = sgd_alpha(logit(0.9), c_black, gt_rgb_empty, 0.0, False, b_black)
trajA_rand  = sgd_alpha(logit(0.9), c_black, gt_rgb_empty, 0.0, True, b_black)
trajB_fixed = sgd_alpha(logit(0.1), g, gt_rgb_wall, 1.0, False, g)
trajB_rand  = sgd_alpha(logit(0.1), g, gt_rgb_wall, 1.0, True, g)
print(f"A(빈 공간) 최종 alpha | 고정: {trajA_fixed[-1]:.3f}  랜덤: {trajA_rand[-1]:.3f}")
print(f"B(회색 벽) 최종 alpha | 고정: {trajB_fixed[-1]:.3f}  랜덤: {trajB_rand[-1]:.3f}")
# 출력: A(빈 공간) 최종 alpha | 고정: 0.900  랜덤: 0.034
# 출력: B(회색 벽) 최종 alpha | 고정: 0.100  랜덤: 0.927

# %%
fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.08,
    subplot_titles=("A. 빈 공간(RGBA GT), 검은 가우시안 c=0",
                    "B. 회색 벽(RGB GT), 배경색 = 벽색",
                    "SGD: 매 스텝 배경을 새로 뽑을 때 alpha 궤적"),
)
fig.add_trace(go.Scatter(x=alphas, y=loss_fixed_A, mode="lines+markers", name="고정 배경 loss",
                         line=dict(color="#9aa0a6", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=alphas, y=loss_rand_A, mode="lines+markers", name="랜덤 배경 E[loss]",
                         line=dict(color="#d9534f")), row=1, col=1)
fig.add_trace(go.Scatter(x=alphas, y=loss_fixed_B, mode="lines+markers", showlegend=False,
                         line=dict(color="#9aa0a6", dash="dash")), row=1, col=2)
fig.add_trace(go.Scatter(x=alphas, y=loss_rand_B, mode="lines+markers", showlegend=False,
                         line=dict(color="#d9534f")), row=1, col=2)
steps = np.arange(len(trajA_rand))
fig.add_trace(go.Scatter(x=steps, y=trajA_fixed, name="A 고정(b=0)", line=dict(color="#9aa0a6", dash="dash")), row=1, col=3)
fig.add_trace(go.Scatter(x=steps, y=trajA_rand, name="A 랜덤", line=dict(color="#d9534f")), row=1, col=3)
fig.add_trace(go.Scatter(x=steps, y=trajB_fixed, name="B 고정(b=g)", line=dict(color="#5b7fa6", dash="dash")), row=1, col=3)
fig.add_trace(go.Scatter(x=steps, y=trajB_rand, name="B 랜덤", line=dict(color="#2a6f97")), row=1, col=3)
fig.update_xaxes(title_text="alpha", row=1, col=1); fig.update_xaxes(title_text="alpha", row=1, col=2)
fig.update_xaxes(title_text="step", row=1, col=3)
fig.update_yaxes(title_text="MSE", row=1, col=1); fig.update_yaxes(title_text="alpha", range=[0, 1], row=1, col=3)
fig.update_layout(
    title="D7 랜덤 배경: 고정 배경에서는 평평한(구분 불가) loss가, 랜덤 배경에서는 올바른 alpha를 유일하게 선호",
    width=1400, height=460, template="plotly_white", legend=dict(orientation="h", y=-0.2),
)
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 5. 정리
#
# | 상황 | 고정 배경 | 랜덤 배경 $b\sim U[0,1]^3$ |
# |---|---|---|
# | 빈 공간(GT alpha=0)을 배경색 가우시안으로 덮기 | loss 동일 → 못 막음 | $\mathbb{E}[\text{loss}]=a^2/3$ → $a\to0$ |
# | 배경색과 같은 벽을 비워두기(구멍) | loss 동일 → 못 막음 | $(1-a)^2\mathbb{E}[(b-g)^2]$ → $a\to1$ |
#
# 배경색이 매 스텝 바뀌면 "$(1-\text{alpha})\,b$" 항이 **alpha를 식별 가능하게** 만들어,
# 가우시안이 배경색을 흉내내며 빈 공간을 덮는 것도, 배경에 기대어 내용을 비워두는 것도 모두 억제됩니다.
# 추론 시(`self.training == False`)에는 고정색(Viser 기본 배경 `[0.149, 0.165, 0.216]`)을 씁니다.
