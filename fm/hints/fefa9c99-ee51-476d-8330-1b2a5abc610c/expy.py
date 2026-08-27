# %% [markdown]
# # Adam 갱신식 실험 — numpy 10줄 구현으로 확인하기
#
# splatfacto는 모든 파라미터 그룹에 `torch.optim.Adam(lr=그룹별, betas=(0.9, 0.999), eps=1e-15)` 을 씁니다.
#
# $$
# m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
# v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,\qquad
# \theta_t = \theta_{t-1} - \eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon}
# $$
#
# 이 스크립트에서 확인할 것:
# 1. 스칼라 2차함수 최소화에서 SGD vs Adam 궤적
# 2. 첫 스텝 $|\Delta\theta| = \eta$ (그래디언트 크기와 무관)
# 3. 그래디언트를 $10^{-6}$ 배 해도 Adam 스텝은 불변 (단, $\epsilon$ 이 충분히 작을 때만)
# 4. 편향 보정 유무 비교

# %%
# 필요 패키지: numpy, plotly, kaleido (PNG 저장용)
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
# ## 1. Adam을 numpy로 — 딱 10줄
#
# `exp_avg` $=m$, `exp_avg_sq` $=v$ 가 PyTorch 내부 상태 이름입니다.


# %%
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-15, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps, self.bc = lr, beta1, beta2, eps, bias_correction
        self.m = self.v = 0.0
        self.t = 0

    def step(self, theta, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1**self.t) if self.bc else self.m
        v_hat = self.v / (1 - self.b2**self.t) if self.bc else self.v
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def sgd_step(theta, g, lr):
    return theta - lr * g


# %% [markdown]
# ## 2. 스칼라 2차함수 $f(\theta) = \tfrac{1}{2} a\,\theta^2$ 최소화 — SGD vs Adam
#
# $a$ 가 크면 SGD는 $\eta a$ 가 2를 넘는 순간 발산하지만, Adam은 이동량이 $\approx\eta$ 로 묶여 안정적입니다.
# 반대로 $a$ 가 아주 작으면 SGD는 거의 움직이지 않는데, Adam은 여전히 $\eta$ 씩 움직입니다.


# %%
def run(opt_fn, theta0, a, n):
    th, traj = theta0, [theta0]
    for _ in range(n):
        th = opt_fn(th, a * th)  # f'(θ) = aθ
        traj.append(th)
    return np.array(traj)


lr, n = 0.1, 60
trajs = {}
for a in [1.0, 1e-4, 25.0]:
    adam = Adam(lr)
    trajs[(a, "SGD")] = run(lambda th, g: sgd_step(th, g, lr), 3.0, a, n)
    trajs[(a, "Adam")] = run(adam.step, 3.0, a, n)
    print(f"a={a:<6g}  SGD 최종 θ={trajs[(a,'SGD')][-1]: .3e}   Adam 최종 θ={trajs[(a,'Adam')][-1]: .3e}")
# 출력:
# a=1       SGD 최종 θ= 5.391e-03   Adam 최종 θ=-1.524e-01
# a=0.0001  SGD 최종 θ= 2.998e+00   Adam 최종 θ=-1.524e-01
# a=25      SGD 최종 θ= 1.103e+11   Adam 최종 θ=-1.524e-01
# → SGD는 a에 따라 "안 움직임 / 수렴 / 발산"이 갈리지만 Adam은 셋 다 정확히 같은 궤적(스케일 불변). Adam은 lr=0.1 고정이라 0 근처에서 ±0.1~0.15 폭으로 진동.

# %% [markdown]
# ## 3. 첫 스텝 $|\Delta\theta| = \eta$ — 그래디언트가 $10^{-13}$ 이어도
#
# $t=1$: $\hat m_1 = g_1,\ \hat v_1 = g_1^2$ 이므로 $\Delta\theta = -\eta\, g_1/(|g_1|+\epsilon) \approx \mp\eta$.
# splatfacto의 means lr $1.6\times10^{-4}$ 로, 분석 노트에서 관찰된 quats 그래디언트 크기 $10^{-13}$ 까지 넣어 봅니다.

# %%
lr_means = 1.6e-4
print(f"{'g_1':>10s} {'|Δθ| (eps=1e-15)':>18s} {'|Δθ| (eps=1e-8)':>18s}")
for g in [1.0, 1e-3, 1e-6, 1e-9, 1e-13]:
    d15 = abs(Adam(lr_means, eps=1e-15).step(0.0, g))
    d8 = abs(Adam(lr_means, eps=1e-8).step(0.0, g))
    print(f"{g:10.0e} {d15:18.6e} {d8:18.6e}")
# 출력:
#        g_1   |Δθ| (eps=1e-15)    |Δθ| (eps=1e-8)
#      1e+00       1.600000e-04       1.600000e-04
#      1e-03       1.600000e-04       1.599984e-04
#      1e-06       1.600000e-04       1.584158e-04
#      1e-09       1.599998e-04       1.454545e-05
#      1e-13       1.584158e-04       1.599984e-09
# → eps=1e-15 이면 g=1e-13 에서도 |Δθ| ≈ lr (1% 감소). PyTorch 기본 eps=1e-8 이면 g≲1e-8 부터 스텝이 g/eps 에 비례해 붕괴(1e-13 에서 10만 배 축소).

# %% [markdown]
# ## 4. 그래디언트 스케일 $\times 10^{-6}$ 에도 Adam 궤적은 불변 — 표로 확인
#
# 2차함수의 $a$ 를 $10^{-6}$ 배 하면 모든 스텝의 그래디언트가 $10^{-6}$ 배가 됩니다.
# 비율 $\hat m/\sqrt{\hat v}$ 는 스케일에 불변이므로 궤적이 (거의) 같아야 합니다 — $\epsilon$ 이 $\sqrt{\hat v}$ 보다 충분히 작을 때만.

# %%
base = run(Adam(0.1, eps=1e-15).step, 3.0, 1.0, 20)
scaled15 = run(Adam(0.1, eps=1e-15).step, 3.0, 1e-6, 20)
scaled8 = run(Adam(0.1, eps=1e-8).step, 3.0, 1e-6, 20)
print(f"{'t':>3s} {'θ (a=1)':>12s} {'θ (a=1e-6, eps=1e-15)':>24s} {'θ (a=1e-6, eps=1e-8)':>24s}")
for t in [0, 1, 2, 5, 10, 20]:
    print(f"{t:3d} {base[t]:12.6f} {scaled15[t]:24.6f} {scaled8[t]:24.6f}")
print("최대 차이 (eps=1e-15):", np.abs(base - scaled15).max())
print("최대 차이 (eps=1e-8) :", np.abs(base - scaled8).max())
# 출력:
#   t      θ (a=1)    θ (a=1e-6, eps=1e-15)     θ (a=1e-6, eps=1e-8)
#   0     3.000000                 3.000000                 3.000000
#   1     2.900000                 2.900000                 2.900332
#   2     2.800103                 2.800103                 2.800772
#   5     2.501779                 2.501779                 2.503484
#  10     2.014188                 2.014188                 2.017662
#  20     1.119360                 1.119360                 1.126087
# 최대 차이 (eps=1e-15): 6.753795300795673e-10
# 최대 차이 (eps=1e-8) : 0.006726294904589647
# → eps=1e-15 이면 그래디언트가 1e-6 배 작아져도 궤적이 ~1e-9 수준으로 동일(= eps/√v 비율만큼의 차이). eps=1e-8 이면 g≈3e-6 이라 eps 가 √v 의 0.3% 수준 → 스텝이 그만큼 줄어 차이가 누적됨.

# %% [markdown]
# ## 5. 편향 보정 유무 비교
#
# $m_0=v_0=0$ 에서 시작하므로 보정 없이는 $m_1 = 0.1 g$, $\sqrt{v_1} = \sqrt{0.001}\,|g| \approx 0.032|g|$.
# 비율은 $0.1/0.032 \approx 3.16$ → 첫 스텝이 **lr의 약 3.2배**로 과대. 보정하면 정확히 $\eta$.
# 일정한 그래디언트 $g=1$ 을 계속 주면서 스텝 크기의 시간 변화를 봅니다.

# %%
T = 200
steps = {}
for bc in [True, False]:
    ad = Adam(1.0, bias_correction=bc)
    th, out = 0.0, []
    for _ in range(T):
        new = ad.step(th, 1.0)
        out.append(abs(new - th))
        th = new
    steps[bc] = np.array(out)
print("첫 5스텝 |Δθ|/lr  보정 O:", np.round(steps[True][:5], 4))
print("첫 5스텝 |Δθ|/lr  보정 X:", np.round(steps[False][:5], 4))
print("t=200 |Δθ|/lr    보정 O: %.4f  보정 X: %.4f" % (steps[True][-1], steps[False][-1]))
# 출력:
# 첫 5스텝 |Δθ|/lr  보정 O: [1. 1. 1. 1. 1.]
# 첫 5스텝 |Δθ|/lr  보정 X: [3.1623 4.2496 4.9502 5.4416 5.7971]
# t=200 |Δθ|/lr    보정 O: 1.0000  보정 X: 2.3482
# → 보정 없으면 첫 스텝 3.2배로 시작해 t≈12 에서 최대 6.6배까지 커진 뒤 아주 천천히 1로 수렴 (t=1000 에 1.26배, t=3000 에 1.03배).
#   이유: m 은 0.9^t 로 빨리 1에 도달하지만 √v 는 0.999^t 로 느리게 1에 도달 → 그 사이 분모가 작아 과대 스텝. β₂ 가 β₁ 보다 1에 훨씬 가까운 것이 원인.

# %% [markdown]
# ## 6. 시각화 (plotly) — 왼쪽: SGD vs Adam 궤적, 오른쪽: 편향 보정 유무의 스텝 크기

# %%
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("f(θ)=½aθ² 최소화: SGD(점선) vs Adam(실선), lr=0.1",
                    "일정 그래디언트에서 |Δθ|/lr — 편향 보정 유무"),
)
colors = {1.0: "#1f77b4", 1e-4: "#2ca02c", 25.0: "#d62728"}
for (a, name), tr in trajs.items():
    y = np.clip(tr, -4, 4)  # 발산한 SGD(a=25)는 축 범위로 잘라 표시
    fig.add_trace(go.Scatter(
        x=np.arange(len(tr)), y=y, mode="lines", name=f"{name}, a={a:g}",
        line=dict(color=colors[a], dash="dot" if name == "SGD" else "solid", width=2),
    ), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(1, T + 1), y=steps[True], mode="lines",
                         name="편향 보정 O", line=dict(color="#1f77b4", width=2)), row=1, col=2)
fig.add_trace(go.Scatter(x=np.arange(1, T + 1), y=steps[False], mode="lines",
                         name="편향 보정 X", line=dict(color="#ff7f0e", width=2)), row=1, col=2)
fig.update_xaxes(title_text="step", row=1, col=1)
fig.update_yaxes(title_text="θ (±4로 클립)", row=1, col=1)
fig.update_xaxes(title_text="step", type="log", row=1, col=2)
fig.update_yaxes(title_text="|Δθ| / lr", row=1, col=2)
fig.update_layout(width=1100, height=450, template="plotly_white",
                  title="Adam: 그래디언트 스케일 불변성과 편향 보정",
                  legend=dict(orientation="h", y=-0.2))
_show(fig)
fig.write_image("expy.png")  # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png
