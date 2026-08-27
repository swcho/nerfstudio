# %% [markdown]
# # `ExponentialDecayScheduler` 실험
#
# splatfacto `means` 설정: $\eta_0 = 1.6\times10^{-4}$, $\eta_T = 1.6\times10^{-6}$, $T = 30000$.
#
# $$\eta(t) = \eta_0\left(\frac{\eta_T}{\eta_0}\right)^{t/T} = \exp\big((1-\tfrac tT)\ln\eta_0 + \tfrac tT\ln\eta_T\big)$$
#
# 필요 패키지: numpy, plotly, kaleido

# %%
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


lr_init, lr_final, T = 1.6e-4, 1.6e-6, 30000

# %% [markdown]
# ## 1. nerfstudio 코드 그대로 (로그 공간 선형 보간) vs 닫힌 식

# %%
def lr_nerfstudio(step, warmup_steps=0):
    t = np.clip((step - warmup_steps) / (T - warmup_steps), 0, 1)
    return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)


def lr_closed(step):
    return lr_init * (lr_final / lr_init) ** (step / T)


steps = np.arange(0, T + 1)
print("최대 오차:", np.max(np.abs(lr_nerfstudio(steps) - lr_closed(steps))))
for s in [0, 1, 15000, 30000]:
    print(f"step {s:>5}: {lr_nerfstudio(s):.6e}")
# 출력: 최대 오차: 3.2526065174565133e-19
# 출력: step     0: 1.600000e-04
# 출력: step     1: 1.599754e-04
# 출력: step 15000: 1.600000e-05
# 출력: step 30000: 1.600000e-06

# %% [markdown]
# ## 2. 스텝당 공비 $r = (1/100)^{1/30000}$ 과 등비 누적 검증
#
# 등비수열이므로 이웃한 스텝의 비는 어디서나 같아야 하고, $r^{30000} = 1/100$ 이어야 합니다.

# %%
r = (lr_final / lr_init) ** (1 / T)
ratios = lr_nerfstudio(steps[1:]) / lr_nerfstudio(steps[:-1])
print(f"공비 r = {r:.6f}   (= exp(-ln100/30000) = {np.exp(-np.log(100)/T):.6f})")
print(f"이웃 비 최소/최대: {ratios.min():.9f} / {ratios.max():.9f}")
print(f"r**30000 = {r**T:.12f}   (기대값 0.01)")
lr = lr_init
for _ in range(T):
    lr *= r
print(f"30000번 곱한 결과: {lr:.6e}   (기대값 {lr_final:.1e})")
# 출력: 공비 r = 0.999847   (= exp(-ln100/30000) = 0.999847)
# 출력: 이웃 비 최소/최대: 0.999846506 / 0.999846506
# 출력: r**30000 = 0.010000000000   (기대값 0.01)
# 출력: 30000번 곱한 결과: 1.600000e-06   (기대값 1.6e-06)

# %% [markdown]
# ## 3. 반감기
#
# $$t_{1/2} = \frac{\ln 2}{-\ln r} = 30000\cdot\frac{\ln 2}{\ln 100}$$

# %%
t_half = np.log(2) / (-np.log(r))
print(f"반감기 = {t_half:.1f} 스텝,  30000스텝 동안 반감 횟수 = {T / t_half:.2f} (= log2(100) = {np.log2(100):.2f})")
print(f"검증: lr({t_half:.0f}) / lr(0) = {lr_nerfstudio(round(t_half)) / lr_init:.4f}")
# 출력: 반감기 = 4515.4 스텝,  30000스텝 동안 반감 횟수 = 6.64 (= log2(100) = 6.64)
# 출력: 검증: lr(4515) / lr(0) = 0.5000

# %% [markdown]
# ## 4. 로그를 취하면 직선: $\ln\eta(t) = \ln\eta_0 + t\ln r$

# %%
log_lr = np.log(lr_nerfstudio(steps))
slope, intercept = np.polyfit(steps, log_lr, 1)
print(f"기울기 = {slope:.6e} (= ln r = {np.log(r):.6e}),  절편 = {intercept:.4f} (= ln lr_init = {np.log(lr_init):.4f})")
print("직선 잔차 최대:", np.max(np.abs(np.polyval([slope, intercept], steps) - log_lr)))
# 출력: 기울기 = -1.535057e-04 (= ln r = -1.535057e-04),  절편 = -8.7403 (= ln lr_init = -8.7403)
# 출력: 직선 잔차 최대: 3.552713678800501e-15

# %% [markdown]
# ## 5. 선형 / 코사인 / 지수 감쇠 비교 (linear y축 vs log y축)
#
# - 선형: $\eta_0 - (\eta_0-\eta_T)\,t/T$
# - 코사인: $\eta_T + \tfrac12(\eta_0-\eta_T)\big(1+\cos(\pi t/T)\big)$
# - 지수: 위 식. 로그 y축에서만 직선이 되며, 각 자릿수($10^{-4}\to10^{-5}$, $10^{-5}\to10^{-6}$)에 같은 시간(15000스텝)을 씁니다.

# %%
lin = lr_init - (lr_init - lr_final) * steps / T
cos = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * steps / T))
exp_ = lr_nerfstudio(steps)
for name, c in [("선형", lin), ("코사인", cos), ("지수", exp_)]:
    frac = np.mean(c > 1e-5)
    print(f"{name:>4}: lr > 1e-5 인 스텝 비율 = {frac:.3f},  step 15000 값 = {c[15000]:.3e}")
# 출력:   선형: lr > 1e-5 인 스텝 비율 = 0.947,  step 15000 값 = 8.080e-05
# 출력:  코사인: lr > 1e-5 인 스텝 비율 = 0.852,  step 15000 값 = 8.080e-05
# 출력:   지수: lr > 1e-5 인 스텝 비율 = 0.602,  step 15000 값 = 1.600e-05

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("linear y축", "log y축 (지수감쇠 = 직선)"))
colors = {"선형": "#8a8f98", "코사인": "#e07b39", "지수 (ExponentialDecay)": "#2f6fdd"}
for col in (1, 2):
    for name, c in [("선형", lin), ("코사인", cos), ("지수 (ExponentialDecay)", exp_)]:
        fig.add_trace(
            go.Scatter(x=steps[::50], y=c[::50], name=name, mode="lines",
                       line=dict(color=colors[name], width=2), showlegend=(col == 1)),
            row=1, col=col,
        )
    fig.add_vline(x=t_half, line=dict(color="#2f6fdd", dash="dot", width=1), row=1, col=col)
fig.add_annotation(x=t_half, y=np.log10(lr_init / 2), text=f"반감기 ≈ {t_half:.0f}", showarrow=False,
                   xanchor="left", row=1, col=2, font=dict(size=11, color="#2f6fdd"))
fig.update_yaxes(type="log", row=1, col=2)
fig.update_xaxes(title_text="step")
fig.update_yaxes(title_text="learning rate", row=1, col=1)
fig.update_layout(
    title="means lr 스케줄: 1.6e-4 → 1.6e-6, 30000 steps",
    template="plotly_white", width=1000, height=420,
    legend=dict(orientation="h", y=-0.2),
)
_show(fig)
fig.write_image("expy.png", scale=2)  # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png
