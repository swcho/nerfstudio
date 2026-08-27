# %% [markdown]
# # 알파 리셋(`reset_opa`) 따라가기
#
# gsplat `DefaultStrategy.step_post_backward`는 `step % reset_every(3000) == 0`마다
# `reset_opa(value = prune_opa * 2 = 0.2)`를 호출합니다. 실제 파라미터는 **로짓** $\tilde\alpha_i$ 이고
# 렌더에 쓰이는 불투명도는 $\alpha_i = \sigma(\tilde\alpha_i)$ 이므로, 리셋은
#
# $$\tilde\alpha_i \leftarrow \min\big(\tilde\alpha_i,\ \mathrm{logit}(0.2)\big),\qquad \mathrm{logit}(p)=\ln\frac{p}{1-p}$$
#
# 즉 0.2 **위**에 있던 가우시안만 0.2로 끌어내리고, 이미 0.2 아래인 것은 그대로 둡니다. 이 스크립트는
# (1) logit/sigmoid 수치, (2) 랜덤 로짓 벡터에 clamp 적용 전후 분포, (3) "유용한" vs "쓸모없는" 가우시안이
# 리셋 후 261스텝 동안 어떻게 갈라지는지 토이 시뮬레이션으로 보여줍니다.
#
# 필요 패키지: numpy, plotly, kaleido (PNG 저장)

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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


PRUNE_OPA = 0.1                 # splatfacto cull_alpha_thresh
RESET_VALUE = PRUNE_OPA * 2.0   # DefaultStrategy: value = prune_opa * 2.0
RESET_LOGIT = logit(RESET_VALUE)
print(f"logit(0.1)  = {logit(0.1):+.4f}  (prune 임계값)")
print(f"logit(0.2)  = {RESET_LOGIT:+.4f}  (reset clamp 상한)")
print(f"sigmoid(logit(0.2)) = {sigmoid(RESET_LOGIT):.4f}  (역변환 확인)")
# 출력:
# logit(0.1)  = -2.1972  (prune 임계값)
# logit(0.2)  = -1.3863  (reset clamp 상한)
# sigmoid(logit(0.2)) = 0.2000  (역변환 확인)

# %% [markdown]
# ## 1. clamp는 "위에서만" 자른다
#
# 학습이 어느 정도 진행된 상태를 흉내 내어 로짓을 넓게 뿌리고 `torch.clamp(p, max=logit(0.2))`와 같은 연산을 적용합니다.
# 리셋 전 불투명도가 0.2를 넘던 것은 전부 정확히 0.2가 되고, 그 아래는 손대지 않으므로 분포가 0.2에서 잘린 모양이 됩니다.

# %%
rng = np.random.default_rng(0)
N = 20_000
logits_before = rng.normal(loc=1.0, scale=2.5, size=N)          # 다수가 꽤 불투명한 상태
logits_after = np.minimum(logits_before, RESET_LOGIT)          # reset_opa 의 핵심 한 줄
opa_before, opa_after = sigmoid(logits_before), sigmoid(logits_after)

print(f"리셋 전: opacity>0.2 비율 {np.mean(opa_before > RESET_VALUE):.1%}, max {opa_before.max():.3f}")
print(f"리셋 후: opacity>0.2 비율 {np.mean(opa_after > RESET_VALUE + 1e-9):.1%}, max {opa_after.max():.3f}")
print(f"바뀐 가우시안 수: {np.sum(logits_before != logits_after):,} / {N:,}  (0.2 아래였던 {np.sum(opa_before <= RESET_VALUE):,}개는 그대로)")
# 출력:
# 리셋 전: opacity>0.2 비율 83.3%, max 1.000
# 리셋 후: opacity>0.2 비율 0.0%, max 0.200
# 바뀐 가우시안 수: 16,661 / 20,000  (0.2 아래였던 3,339개는 그대로)

# %% [markdown]
# ## 2. 리셋 뒤 261스텝: 회복하는 것과 못 하는 것
#
# nerfstudio는 `pause_refine_after_reset = num_train_data + refine_every` (예: 161 + 100 = 261) 로 설정하여
# 리셋 직후 261스텝 동안은 grow/prune을 **멈춥니다**. 이유는 리셋 직후엔 모든 가우시안이 0.2로 반투명이라
# 어느 것이 진짜 필요한지 아직 모르기 때문입니다. 모든 학습 이미지를 최소 한 번씩(161장) 다시 보고 나서
# 다음 refine 스텝(100의 배수)에 도달할 때까지 기다리면, 그 사이에
#
# - **유용한** 가우시안: 화면에 보이는 매 스텝 재구성 손실이 "더 불투명해져라"는 그래디언트를 주어 로짓이 다시 올라간다.
# - **쓸모없는** 가우시안(플로터, 가려진 것, 배경과 겹친 것): 그래디언트가 거의 0 → 로짓이 0.2 근처에 머문다.
#   여기에 리셋과 함께 **Adam 모멘트 `exp_avg`, `exp_avg_sq`가 0으로 초기화**되므로 과거의 관성으로 다시 불투명해질 수도 없다.
#
# 아래 시뮬레이션은 Adam 업데이트를 단순화해 두 부류를 비교합니다. 주의: 실제 `_prune_gs`는 `sigmoid(opacity) < 0.1`을 지우므로
# 0.2에 머무는 것은 "prune 임계 아래"가 아니라 "여전히 반투명, 더 이상 못 올라옴" 상태이고, 리셋 전에 이미 0.2보다 낮았던 것,
# 그리고 261스텝 동안 아주 약한 음의 그래디언트라도 받은 것은 0.1 아래로 떨어져 잘려 나갑니다.

# %%
STEPS = 400          # 리셋 후 관찰 스텝 수
PAUSE = 261          # pause_refine_after_reset
lr = 0.05            # splatfacto opacities lr
b1, b2, eps = 0.9, 0.999, 1e-8


def adam_traj(grad_fn, x0, reset_adam=True):
    x = x0
    m = 0.0 if reset_adam else -0.3    # 리셋 없으면 예전 관성이 남아 있는 상황 흉내
    v = 0.0 if reset_adam else 0.09
    xs = [x]
    for t in range(1, STEPS + 1):
        g = grad_fn(t)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1**t), v / (1 - b2**t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        xs.append(x)
    return np.array(xs)


rng = np.random.default_rng(1)
# 유용한 가우시안: 보이는 스텝(대략 절반)에는 "더 불투명하게" (음의 grad → 로짓 상승), 나머지는 0
useful_grad = lambda t: (-1.0 + 0.3 * rng.normal()) if rng.random() < 0.5 else 0.0
# 쓸모없는 가우시안(플로터): 거의 0, 가끔 "더 투명하게" (양의 grad → 로짓 하강)
floater_grad = lambda t: (+0.4 * abs(rng.normal())) if rng.random() < 0.08 else 0.0

traj_useful = sigmoid(adam_traj(useful_grad, RESET_LOGIT))
traj_floater = sigmoid(adam_traj(floater_grad, RESET_LOGIT))
traj_floater_noreset = sigmoid(adam_traj(floater_grad, RESET_LOGIT, reset_adam=False))

print(f"step {PAUSE} (첫 refine) 시점 opacity — 유용: {traj_useful[PAUSE]:.3f} | 플로터(Adam 리셋): {traj_floater[PAUSE]:.3f} | 플로터(Adam 리셋 안 함): {traj_floater_noreset[PAUSE]:.3f}")
print(f"플로터가 prune 임계(0.1) 아래로 떨어지는 첫 스텝: {int(np.argmax(traj_floater < PRUNE_OPA))}")
# 출력:
# step 261 (첫 refine) 시점 opacity — 유용: 1.000 | 플로터(Adam 리셋): 0.010 | 플로터(Adam 리셋 안 함): 0.168
# 플로터가 prune 임계(0.1) 아래로 떨어지는 첫 스텝: 113

# %% [markdown]
# `exp_avg`를 0으로 만들지 않으면(3번째 곡선) 리셋 직전까지 "불투명해져라"고 쌓여 있던 모멘트가
# 리셋 직후에도 계속 로짓을 밀어올려 플로터가 그대로 복귀합니다. Adam 상태를 함께 지우는 것이
# 리셋을 실제로 "리셋"이 되게 만드는 핵심입니다.

# %%
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.45, 0.55],
    subplot_titles=("reset_opa 전/후 불투명도 분포 (N=20,000)",
                    "리셋 후 궤적: 유용한 가우시안 vs 플로터"),
)
fig.add_trace(go.Histogram(x=opa_before, nbinsx=50, name="리셋 전", opacity=0.55, marker_color="#8c8c8c"), row=1, col=1)
fig.add_trace(go.Histogram(x=opa_after, nbinsx=50, name="리셋 후 (≤0.2)", opacity=0.75, marker_color="#1f77b4"), row=1, col=1)
fig.add_vline(x=RESET_VALUE, line_dash="dash", line_color="#1f77b4", row=1, col=1,
              annotation_text="0.2 = prune_opa×2", annotation_position="top right")
fig.add_vline(x=PRUNE_OPA, line_dash="dot", line_color="#d62728", row=1, col=1,
              annotation_text="prune 0.1", annotation_position="top left")

steps = np.arange(STEPS + 1)
fig.add_trace(go.Scatter(x=steps, y=traj_useful, name="유용 (grad 받음)", line=dict(color="#2ca02c", width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=steps, y=traj_floater, name="플로터 (Adam 리셋)", line=dict(color="#d62728", width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=steps, y=traj_floater_noreset, name="플로터 (Adam 리셋 안 함)", line=dict(color="#ff7f0e", dash="dash")), row=1, col=2)
fig.add_hline(y=PRUNE_OPA, line_dash="dot", line_color="#d62728", row=1, col=2,
              annotation_text="prune 임계 0.1", annotation_position="bottom right")
fig.add_vrect(x0=0, x1=PAUSE, fillcolor="#999999", opacity=0.15, line_width=0, row=1, col=2,
              annotation_text="pause_refine_after_reset = 261 (grow/prune 정지)", annotation_position="top left")
fig.add_vline(x=PAUSE, line_dash="dash", line_color="black", row=1, col=2)

fig.update_xaxes(title_text="opacity = sigmoid(logit)", row=1, col=1)
fig.update_yaxes(title_text="count", row=1, col=1)
fig.update_xaxes(title_text="리셋 후 step", row=1, col=2)
fig.update_yaxes(title_text="opacity", range=[-0.02, 1.02], row=1, col=2)
fig.update_layout(barmode="overlay", width=1250, height=480, template="plotly_white",
                  legend=dict(orientation="h", y=-0.18))
_show(fig)
fig.write_image("expy.png", scale=2)   # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png
