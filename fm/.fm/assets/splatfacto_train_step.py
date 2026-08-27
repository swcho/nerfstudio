# %% [markdown]
# # Splatfacto 학습 스텝 완전 분해 — `trainer.train_iteration()` 한 번을 한 줄씩
#
# [splatfacto_walkthrough.py](splatfacto_walkthrough.py) ④에서는 `get_outputs()` / `get_loss_dict()` / `backward()`를 **함수 단위**로 호출했습니다.
# 여기서는 그 함수들 **안쪽**까지 펼쳐서, nerfstudio·gsplat 코드를 거치지 않고 같은 계산을 직접 수행하고 원본과 결과가 일치하는지 확인합니다.
#
# `Trainer.train()` 루프 한 바퀴 ([trainer.py:246-273](../nerfstudio/engine/trainer.py#L246-L273)) 와 `train_iteration()` ([trainer.py:487-530](../nerfstudio/engine/trainer.py#L487-L530)) 을 합치면:
#
# ```
# A. BEFORE 콜백      model.step_cb(optimizers, step)         step 갱신, optimizers 주입
# B. zero_grad        optimizers.zero_grad_some(...)
# C. 데이터            datamanager.next_train(step)            (camera, {"image"})
# D. forward          model.get_outputs(camera)               카메라 보정 → 해상도 스케줄 → viewmat/K → rasterization → 배경 합성
# E. metrics/loss     model.get_metrics_dict / get_loss_dict   PSNR / 0.8·L1 + 0.2·(1−SSIM)
# F. backward         grad_scaler.scale(loss).backward()
# G. optimizer step   optimizers.optimizer_scaler_step_some    6개 Adam 각각 step
# H. scheduler        optimizers.scheduler_step_all            means lr 지수감쇠
# I. AFTER 콜백       model.step_post_backward(step)          strategy.step_post_backward: 통계 누적 → (100스텝마다) 분할/복제/컬링 → (3000마다) 알파 리셋
# ```
#
# 아래 셀들이 A~I를 순서대로 하나씩 맡습니다. 마지막에 이걸 함수 하나로 묶어 700스텝을 돌려 **첫 densification이 optimizer 상태까지 어떻게 바꾸는지** 봅니다.

# %%
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _find_repo() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists() and (d / "nerfstudio").is_dir():
            return d
    raise RuntimeError(f"nerfstudio 저장소 루트를 못 찾음 (시작점 {start})")


REPO = _find_repo()
os.chdir(REPO)
DATA = REPO / "data/mipnerf360/garden"
assert DATA.exists(), f"{DATA} 없음 — `dm use mipnerf360@2.0.0 --link data/mipnerf360` 먼저"
device = torch.device("cuda")

import matplotlib.font_manager as fm

_ko = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("Noto Sans CJK", "Noto Sans KR", "Nanum", "Malgun"))]
if _ko:
    plt.rcParams["font.family"] = sorted(set(_ko))[0]
plt.rcParams["axes.unicode_minus"] = False

# --- walkthrough ②와 동일한 셋업 (압축) ---
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline

config = copy.deepcopy(method_configs["splatfacto"])
assert isinstance(config, TrainerConfig)
config.pipeline.datamanager.dataparser = ColmapDataParserConfig(
    data=DATA, colmap_path=Path("sparse/0"), downscale_factor=4, downscale_rounding_mode="round", load_3D_points=True
)
config.experiment_name = "garden-train-step"
config.output_dir = REPO / "outputs"
config.vis = "tensorboard"
config.logging.local_writer.enable = False
config.set_timestamp()

# InstantiateConfig.setup() 은 -> Any 라 타입이 죽음. TrainerConfig._target == Trainer 이므로 명시.
trainer: Trainer = config.setup(local_rank=0, world_size=1)
trainer.setup(test_mode="test")
pipeline = trainer.pipeline
assert isinstance(pipeline, VanillaPipeline)
model = pipeline.model
assert isinstance(model, SplatfactoModel)
dm = pipeline.datamanager
assert isinstance(dm, FullImageDatamanager)
opt = trainer.optimizers  # nerfstudio.engine.optimizers.Optimizers — 파라미터 그룹별 Adam + 스케줄러 묶음
print(f"가우시안 {model.num_points:,}개 | optimizer 그룹: {list(opt.optimizers)} | 스케줄러: {list(opt.schedulers)}")
print("mixed_precision:", config.mixed_precision, "→ GradScaler enabled =", trainer.grad_scaler.is_enabled())


# %% [markdown]
# ## 0. optimizer 설정 — 8개 선언, 6개 생성
#
# 위 출력에서 `optimizer 그룹`이 6개, `스케줄러`가 1개만 찍힙니다. 왜 그런지 보려면 optimizer가 **어디서 어떻게 만들어지는지**부터 봐야 합니다.
#
# nerfstudio는 모델 전체에 optimizer 하나를 두지 않습니다. **파라미터 그룹마다 독립적인 Adam + 스케줄러**를 붙이고, 두 축을 그룹 이름(문자열)으로 join합니다:
#
# ```
# TrainerConfig.optimizers: Dict[group_name, {"optimizer": OptimizerConfig, "scheduler": SchedulerConfig | None}]
#                                   │            ← method_configs.py:607-644 에 선언
#                                   ├── 그룹 이름으로 join (Optimizers.__init__, optimizers.py:82-114)
#                                   │
# Model.get_param_groups() -> Dict[group_name, List[Parameter]]   ← splatfacto.py:420-430 이 결정
# ```
#
# 모델이 내놓은 그룹이 config에 없으면 그 자리에서 `RuntimeError`. 반대로 **config에만 있고 모델이 안 내놓는 그룹은 조용히 무시**됩니다 — 이게 8 vs 6의 정체입니다.
#
# ### splatfacto의 파라미터 그룹
#
# | 그룹 | 최적화 대상 | shape | lr | scheduler |
# |---|---|---|---|---|
# | `means` | 가우시안 3D 중심 좌표 | (N, 3) | `1.6e-4` | ExpDecay → `1.6e-6` @30k |
# | `scales` | log 스케일 (축별 크기) | (N, 3) | `5e-3` | 없음 (상수) |
# | `quats` | 회전 쿼터니언 | (N, 4) | `1e-3` | 없음 |
# | `opacities` | logit 불투명도 | (N, 1) | `5e-2` | 없음 |
# | `features_dc` | SH 0차 = 시점 독립 기본 색 | (N, 3) | `2.5e-3` | 없음 |
# | `features_rest` | SH 1~3차 = 시점 의존 색 | (N, 15, 3) | `2.5e-3 / 20` = `1.25e-4` | 없음 |
# | ~~`camera_opt`~~ | 카메라 pose 보정 | (n_cams, 6) | `1e-4` | warmup 1k → `5e-7` @30k |
# | ~~`bilateral_grid`~~ | 이미지별 ISP 보정 격자 | (n_cams,16,16,8,12) | `2e-3` | warmup 1k → `1e-4` @30k |
#
# 전부 `AdamOptimizerConfig(eps=1e-15)`. 취소선 두 개는 조건부라 기본 실행에서 **생성되지 않습니다**:
#
# - `camera_opt` → `camera_optimizer.mode != "off"` 일 때만 ([camera_optimizers.py:200-207](../nerfstudio/cameras/camera_optimizers.py#L200-L207)). splatfacto 기본값이 `"off"`라서 D1 단계에서 보정량이 0으로 나옵니다.
# - `bilateral_grid` → `use_bilateral_grid=True` 일 때만 (기본 `False`).
#
# lr 값은 3DGS 원논문 설정을 그대로 가져온 것입니다. 두 가지가 눈에 띕니다:
#
# - **`features_rest`가 `features_dc`의 1/20** — 고차 SH는 시점 의존 성분이라 빨리 움직이면 floater/과적합이 생깁니다. (G단계 표에서 이 둘의 Δ 차이로 확인됩니다.)
# - **`means`에만 스케줄러** — 초반엔 SfM 포인트를 정렬하느라 크게 움직여야 하고, 후반엔 고정되어야 densification이 수렴합니다. H단계 그래프가 이 곡선입니다.
#
# ### OptimizerConfig 필드 ([optimizers.py:33-71](../nerfstudio/engine/optimizers.py#L33-L71))
#
# | 필드 | 의미 |
# |---|---|
# | `lr` | 학습률. **스케줄러의 `lr_init`으로도 쓰인다** (아래 참조) |
# | `eps` | Adam epsilon. splatfacto는 `1e-15` — G단계 갱신식의 $\epsilon$ |
# | `weight_decay` | `AdamOptimizerConfig` / `RAdamOptimizerConfig`에만 존재 |
# | `max_norm` | grad clipping 임계값. splatfacto는 전부 `None` |
#
# `setup()`은 `_target`과 `max_norm`만 빼고 **나머지 필드를 그대로 `**kwargs`로** torch optimizer에 넘깁니다. 그래서 dataclass에 필드를 추가하면 자동 전달되지만, torch가 모르는 이름을 넣으면 `TypeError`가 납니다. `max_norm`만 예외적으로 torch에 안 넘기고 nerfstudio가 step 직전에 직접 `clip_grad_norm_`을 겁니다 ([optimizers.py:159-172](../nerfstudio/engine/optimizers.py#L159-L172) — G단계에서 호출).
#
# ### Scheduler 설정 ([schedulers.py:92-137](../nerfstudio/engine/schedulers.py#L92-L137))
#
# | 필드 | 기본 | 의미 |
# |---|---|---|
# | `lr_pre_warmup` | `1e-8` | warmup 시작 lr |
# | `lr_final` | `None` | 종료 lr. None이면 `lr_init` (= 감쇠 없음) |
# | `warmup_steps` | `0` | warmup 구간 길이 |
# | `max_steps` | `100000` | 이 스텝에서 `lr_final` 도달 |
# | `ramp` | `"cosine"` | warmup 곡선 (`"linear"` 가능) |
#
# 시작 lr(`lr_init`)은 스케줄러 config가 아니라 **짝지어진 optimizer의 `lr`** 에서 읽습니다 ([optimizers.py:105](../nerfstudio/engine/optimizers.py#L105)). 그래서 optimizer의 `lr`만 바꿔도 감쇠 스케줄 전체가 따라 스케일됩니다.
#
# ⚠️ **`max_steps`는 `TrainerConfig.max_num_iterations`와 자동 동기화되지 않습니다.** `--max-num-iterations 10000`으로 줄이면 `means` lr은 `1.6e-6`까지 못 내려가고 중간(약 `7e-6`)에서 학습이 끝납니다. 스텝 수를 바꿀 땐 스케줄러 `max_steps`도 같이 넘겨야 합니다.
#
# ### CLI 오버라이드
#
# 언더스코어는 대시로 바뀝니다:
#
# ```bash
# ns-train splatfacto --data DATA \
#   --optimizers.means.optimizer.lr 1e-4 \
#   --optimizers.means.scheduler.lr-final 1e-6 \
#   --optimizers.means.scheduler.max-steps 10000 \
#   --optimizers.features-dc.optimizer.lr 0.01 \
#   --optimizers.camera-opt.optimizer.lr 1e-3 \
#   --pipeline.model.camera-optimizer.mode SO3xR3   # ← 이게 없으면 camera_opt 그룹 자체가 안 생김
# ```
#
# 최종 해석된 값은 `outputs/<exp>/<method>/<timestamp>/config.yml`에 그대로 덤프됩니다.
#
# ### 프리셋 3개 차이
#
# `splatfacto` / `splatfacto-big` / `splatfacto-mcmc`의 **optimizers 블록은 완전히 동일**합니다. 차이는 전부 `SplatfactoModelConfig` 쪽:
#
# | | 모델 config 차이 |
# |---|---|
# | `splatfacto` | 기본값 |
# | `splatfacto-big` | `cull_alpha_thresh=0.005`, `densify_grad_thresh=0.0005` (가우시안을 더 많이 남김) |
# | `splatfacto-mcmc` | `strategy="mcmc"`, `cull_alpha_thresh=0.005`, `stop_split_at=25000` |
#
# MCMC를 쓰면 I단계에서 `self.schedulers["means"].get_last_lr()[0]`을 읽어가 **현재 means lr에 비례하는 노이즈를 좌표에 주입**합니다 ([splatfacto.py:365-385](../nerfstudio/models/splatfacto.py#L365-L385)). 즉 `means` lr을 만지면 학습률과 샘플링 노이즈 세기가 동시에 변합니다.

# %%
declared = config.optimizers
created = set(opt.optimizers)

print(f"{'group':16s} {'생성':^6s} {'lr':>10s} {'eps':>8s} {'max_norm':>9s}  scheduler")
for name, cfg in declared.items():
    o, s = cfg["optimizer"], cfg["scheduler"]
    sch = "None (상수 lr)" if s is None else (
        f"{type(s).__name__[: -len('Config')]}(lr_final={s.lr_final:.1e}, max_steps={s.max_steps}, warmup={s.warmup_steps})"
    )
    print(f"{name:16s} {'O' if name in created else '·':^6s} {o.lr:10.2e} {o.eps:8.0e} {str(o.max_norm):>9s}  {sch}")

missing = sorted(set(declared) - created)
print(f"\n선언 {len(declared)}개 / 실제 생성 {len(created)}개 — {missing} 는 모델이 파라미터를 안 내놓아 생성되지 않음")
print("  camera_optimizer.mode =", config.pipeline.model.camera_optimizer.mode, "| use_bilateral_grid =", config.pipeline.model.use_bilateral_grid)

# 스케줄러의 시작 lr은 scheduler config가 아니라 짝지어진 optimizer의 lr에서 온다
print(f"\nmeans: optimizer.lr = {declared['means']['optimizer'].lr:.2e} → 스케줄러 현재 lr = {opt.schedulers['means'].get_last_lr()[0]:.2e}  (일치)")
print(f"max_num_iterations = {config.max_num_iterations} vs means scheduler.max_steps = {declared['means']['scheduler'].max_steps}"
      "  ← 둘은 자동 동기화되지 않는다")

# gradient accumulation은 그룹마다 다른 주기를 줄 수 있다 (config 기본 {} → defaultdict가 전부 1로 채움)
print("\ngradient_accumulation_steps:", {g: trainer.gradient_accumulation_steps[g] for g in opt.parameters})


# %% [markdown]
# ## A. BEFORE 콜백 — `model.step_cb` ([splatfacto.py:407-410](../nerfstudio/models/splatfacto.py#L407-L410))
#
# Trainer는 모델에 optimizer를 직접 넘기지 않습니다. 대신 매 스텝 시작에 콜백으로 `step`과 함께 `optimizers`/`schedulers`를 꽂아 줍니다.
# 이게 필요한 이유는 I단계에서 보입니다 — densification이 가우시안 개수를 바꾸면 **Adam의 모멘트 텐서(`exp_avg`, `exp_avg_sq`)도 같은 크기로 늘려야** 하기 때문입니다.

# %%
from nerfstudio.engine.callbacks import TrainingCallbackLocation

STEP = 0
pipeline.train()
print("콜백 전: hasattr(model, 'optimizers') =", hasattr(model, "optimizers"))
for cb in trainer.callbacks:
    cb.run_callback_at_location(STEP, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
print("콜백 후: model.step =", model.step, "| model.optimizers 그룹 =", list(model.optimizers))
print("(model.optimizers is opt.optimizers) =", model.optimizers is opt.optimizers, " ← 같은 객체를 공유")


# %% [markdown]
# ## B. zero_grad — `optimizers.zero_grad_some(needs_zero)` ([trainer.py:494-497](../nerfstudio/engine/trainer.py#L494-L497))
#
# gradient accumulation을 지원하기 위해 그룹별로 "이번 스텝에 0으로 만들 그룹"을 고르지만, splatfacto 기본은 accumulation 1이라 **매 스텝 전부** 0으로 만듭니다.

# %%
needs_zero = [g for g in opt.parameters if STEP % trainer.gradient_accumulation_steps[g] == 0]
opt.zero_grad_some(needs_zero)
print("zero_grad 대상:", needs_zero)
print("means.grad is None:", model.means.grad is None)


# %% [markdown]
# ## C. 데이터 — `datamanager.next_train(step)`
#
# 캐시된 161장 중 **아직 이번 epoch에서 안 본 것** 하나를 꺼냅니다 (`train_unseen_cameras` 리스트를 pop; 다 쓰면 다시 섞음).
# 반환은 `(Cameras[1], {"image": uint8 [H,W,3]})`. 카메라에는 `metadata["cam_idx"]`가 붙어 카메라 옵티마이저(꺼져 있음)가 어떤 포즈를 보정할지 알 수 있습니다.

# %%
camera, batch = dm.next_train(STEP)
gt_u8 = batch["image"]
print(f"cam_idx={camera.metadata['cam_idx']} | 남은 unseen 카메라 {len(dm.train_unseen_cameras)} / {len(dm.train_dataset)}")
print(f"image: {tuple(gt_u8.shape)} {gt_u8.dtype} on {gt_u8.device} | fx={camera.fx.item():.1f} cx={camera.cx.item():.1f} W×H={camera.width.item()}×{camera.height.item()}")
print("camera_to_worlds [3,4]:\n", camera.camera_to_worlds[0].cpu().numpy().round(3))


# %% [markdown]
# ## D. forward — `model.get_outputs(camera)` 를 펼치면 ([splatfacto.py:485-600](../nerfstudio/models/splatfacto.py#L485-L600))
#
# ```
# D1. 카메라 보정      optimized_c2w = camera_optimizer.apply_to_camera(camera)     (mode="off" → 그대로)
# D2. 해상도 스케줄    d = 2^max(num_downscales − step//resolution_schedule, 0)       step 0 → 4
# D3. 뷰 행렬/내부행렬  viewmat = get_viewmat(c2w) [1,4,4],  K [1,3,3] (d로 축소)
# D4. SH 차수          sh_degree_to_use = min(step // 1000, 3)                        step 0 → 0
# D5. 래스터화         gsplat.rasterization(means, quats, exp(scales), sigmoid(opacities), SH계수, viewmat, K, W, H, ...)
#                      → render [1,H,W,3], alpha [1,H,W,1], info{radii, means2d, ...}
# D6. pre_backward    strategy.step_pre_backward(...)   means2d.retain_grad() — backward 때 화면공간 그래디언트를 남기게
# D7. 배경 합성        rgb = render + (1 − alpha) · background(랜덤)
# ```
#
# **D3 — 뷰 행렬.** 카메라 포즈 $c2w = [R \mid t]$ 를 뒤집어 world→camera 변환을 만들고, nerfstudio(OpenGL, $-z$ 전방)와 gsplat(OpenCV, $+z$ 전방) 규약 차이를 $y, z$ 축 부호 반전으로 맞춥니다:
#
# $$
# \text{viewmat} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix},
# \qquad
# K = \frac{1}{d}\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & d \end{bmatrix}
# $$
#
# **D5 — 래스터화.** 가우시안 $i$ 의 3D 공분산은 스케일 $s_i = \exp(\tilde s_i)$ 와 회전 $R(q_i)$ 로 $\Sigma_i = R S S^\top R^\top$ 이고, 화면에 투영된 2D 가우시안 $\mathcal N(\mu_i^{2D}, \Sigma_i^{2D})$ 로 픽셀 $p$ 에 기여합니다.
# 픽셀 색은 깊이 순 알파 블렌딩입니다 ($\alpha_i = \sigma(\tilde\alpha_i)\cdot \exp\!\big(-\tfrac12 (p-\mu_i)^\top (\Sigma_i^{2D})^{-1} (p-\mu_i)\big)$, $c_i$ 는 SH에서 뷰 방향으로 평가한 색):
#
# $$
# \text{render}(p) = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j),
# \qquad
# \text{alpha}(p) = 1 - \prod_i (1-\alpha_i)
# $$
#
# **D7 — 배경 합성.** 투과율이 남은 곳( $1-\text{alpha}$ )에 랜덤 배경색 $b \sim U[0,1]^3$ 을 채웁니다. 매 스텝 배경이 바뀌므로 가우시안이 "배경색을 흉내내는" 식으로 빈 공간을 덮는 것이 억제됩니다:
#
# $$
# \text{rgb}(p) = \operatorname{clamp}\big(\text{render}(p) + (1-\text{alpha}(p))\,b,\ 0,\ 1\big)
# $$
#
# 아래에서 D1~D7을 직접 실행하고, 같은 시드로 `model.get_outputs()`를 호출해 픽셀이 일치하는지 봅니다.

# %%
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from nerfstudio.models.splatfacto import get_viewmat

assert isinstance(model.strategy, DefaultStrategy)

# D1
optimized_c2w = model.camera_optimizer.apply_to_camera(camera)  # [1,3,4]
print("D1 camera_optimizer mode =", model.config.camera_optimizer.mode, "→ 보정량 =", (optimized_c2w - camera.camera_to_worlds).abs().max().item())

# D2
d = 2 ** max(model.config.num_downscales - STEP // model.config.resolution_schedule, 0)
print(f"D2 downscale d = 2^max({model.config.num_downscales} - {STEP}//{model.config.resolution_schedule}, 0) = {d}")

# D3 — 카메라를 잠시 1/d로 줄여 K, W, H를 뽑고 원복
camera.rescale_output_resolution(1 / d)
viewmat = get_viewmat(optimized_c2w)                     # [1,4,4]  world → camera (gsplat 규약: y,z 축 뒤집음)
K = camera.get_intrinsics_matrices().cuda()              # [1,3,3]
W, H = int(camera.width.item()), int(camera.height.item())
camera.rescale_output_resolution(d)
print(f"D3 학습 해상도 {W}×{H} | K[0]=\n{K[0].cpu().numpy().round(1)}")
print("   viewmat[0] (= [R|t] of world→cam, z/y flipped):\n", viewmat[0].detach().cpu().numpy().round(3))

# D4
sh_degree_to_use = min(STEP // model.config.sh_degree_interval, model.config.sh_degree)
colors_sh = torch.cat((model.features_dc[:, None, :], model.features_rest), dim=1)  # [N,16,3]
print(f"D4 SH degree {sh_degree_to_use} 사용 (계수 텐서 {tuple(colors_sh.shape)} 중 앞 {(sh_degree_to_use+1)**2}개 밴드만 유효)")

# D5 — gsplat 호출. 활성함수는 여기서 적용: exp(scales), sigmoid(opacities)
torch.manual_seed(123)  # 배경색 비교를 위해 (D7의 torch.rand)
render, alpha, info = rasterization(
    means=model.means, quats=model.quats, scales=torch.exp(model.scales), opacities=torch.sigmoid(model.opacities).squeeze(-1),
    colors=colors_sh, viewmats=viewmat, Ks=K, width=W, height=H,
    packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=sh_degree_to_use, sparse_grad=False,
    absgrad=model.strategy.absgrad, rasterize_mode=model.config.rasterize_mode,
)
print(f"D5 render {tuple(render.shape)} alpha {tuple(alpha.shape)} | info keys: {sorted(k for k in info if torch.is_tensor(info[k]))}")
print(f"   보이는 가우시안(radii>0): {(info['radii'] > 0).sum().item():,} / {model.num_points:,}")

# D6
model.strategy.step_pre_backward(model.gauss_params, model.optimizers, model.strategy_state, STEP, info)
print("D6 means2d.requires_grad =", info["means2d"].requires_grad, "| retains_grad =", info["means2d"].retains_grad)

# D7
background = torch.rand(3, device=device)
rgb_manual = torch.clamp(render[..., :3] + (1 - alpha) * background, 0.0, 1.0).squeeze(0)  # [H,W,3]
print("D7 background =", background.tolist())

# --- 검증: 원본 get_outputs와 픽셀 단위 비교 (같은 시드 → 같은 배경색) ---
torch.manual_seed(123)
outputs = model.get_outputs(camera)
print(f"\n검증 | max|rgb_manual − get_outputs.rgb| = {(rgb_manual - outputs['rgb']).abs().max().item():.2e}   background 일치: {torch.allclose(background, outputs['background'])}")
info = model.info  # 이후 단계는 원본 호출의 info를 사용 (retain_grad 훅이 걸린 쪽)

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].imshow(rgb_manual.detach().cpu()), ax[0].set_title(f"D5+D7 직접 계산 ({W}×{H})")
ax[1].imshow(alpha.squeeze().detach().cpu(), cmap="gray", vmin=0, vmax=1), ax[1].set_title("alpha (accumulation)")
ax[2].imshow(((1 - alpha) * background).squeeze(0).detach().cpu()), ax[2].set_title("(1−alpha)·background — 구멍에 채워진 랜덤 배경")
[a.axis("off") for a in ax]
plt.show()


# %% [markdown]
# ## E. metrics / loss — `get_metrics_dict` + `get_loss_dict` 를 펼치면 ([splatfacto.py:631-689](../nerfstudio/models/splatfacto.py#L631-L689))
#
# ```
# gt = downscale(image/255, d)                       학습 해상도로 축소 (평균 풀링)
# gt = composite_with_background(gt, background)     RGBA 이미지일 때만 의미 있음 (여기는 RGB → 그대로)
# L1   = mean|gt − rgb|
# SSIM = torchmetrics SSIM(kernel 11, data_range 1)
# main_loss = (1 − λ)·L1 + λ·(1 − SSIM),  λ = ssim_lambda = 0.2
# ```
#
# 학습 해상도의 GT $I$ 와 렌더 $\hat I$ (모두 $[H,W,3]$, $[0,1]$) 에 대해:
#
# $$
# \mathcal L_1 = \frac{1}{3HW}\sum_{p,c}\big|I_{p,c} - \hat I_{p,c}\big|,
# \qquad
# \mathcal L = (1-\lambda)\,\mathcal L_1 + \lambda\,\big(1 - \mathrm{SSIM}(I,\hat I)\big),\quad \lambda = 0.2
# $$
#
# SSIM은 $11\times11$ 가우시안 윈도우로 국소 평균 $\mu$, 분산 $\sigma^2$, 공분산 $\sigma_{I\hat I}$ 를 구해 윈도우별 값을 평균한 것입니다 ($C_1 = (0.01)^2,\ C_2 = (0.03)^2$, data_range 1):
#
# $$
# \mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2 + C_1)(\sigma_x^2+\sigma_y^2 + C_2)}
# $$
#
# $\mathcal L_1$ 은 픽셀 단위 색 오차를, $1-\mathrm{SSIM}$ 은 국소 구조(대비·질감) 오차를 벌점으로 줍니다. 모니터링용 PSNR은 MSE 기준입니다:
#
# $$
# \mathrm{PSNR} = -10\log_{10}\Big(\frac{1}{3HW}\sum_{p,c}(I_{p,c}-\hat I_{p,c})^2\Big)
# $$

# %%
from nerfstudio.models.splatfacto import resize_image

gt = gt_u8.float() / 255.0
gt = resize_image(gt, d) if d > 1 else gt                     # get_gt_img → _downscale_if_required
gt = model.composite_with_background(gt, background)         # RGB라 no-op
pred = outputs["rgb"]

L1 = (gt - pred).abs().mean()
ssim = model.ssim(gt.permute(2, 0, 1)[None], pred.permute(2, 0, 1)[None])
lam = model.config.ssim_lambda
main_loss_manual = (1 - lam) * L1 + lam * (1 - ssim)
psnr_manual = -10 * torch.log10(((gt - pred) ** 2).mean())

metrics_dict = model.get_metrics_dict(outputs, batch)
loss_dict = model.get_loss_dict(outputs, batch, metrics_dict)
print(f"L1 = {L1.item():.5f}   SSIM = {ssim.item():.4f}   → main_loss = 0.8·{L1.item():.4f} + 0.2·{1-ssim.item():.4f} = {main_loss_manual.item():.5f}")
print("원본 get_loss_dict:", {k: round(v.item(), 5) for k, v in loss_dict.items()})
print(f"원본 get_metrics_dict psnr = {metrics_dict['psnr'].item():.4f}  (직접 계산 {psnr_manual.item():.4f}) | gaussian_count = {int(metrics_dict['gaussian_count'])}")
assert torch.allclose(main_loss_manual, loss_dict["main_loss"], atol=1e-6)

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].imshow(gt.cpu()), ax[0].set_title(f"GT (1/{d} 해상도, 배경 합성 후)")
ax[1].imshow(pred.detach().cpu()), ax[1].set_title("pred")
ax[2].imshow((gt - pred).abs().mean(-1).detach().cpu(), cmap="magma"), ax[2].set_title("|GT − pred| — 하늘/배경(포인트 없는 곳)이 가장 크다")
[a.axis("off") for a in ax]
plt.show()


# %% [markdown]
# ## F. backward — `grad_scaler.scale(loss).backward()` ([trainer.py:503-504](../nerfstudio/engine/trainer.py#L503-L504))
#
# `mixed_precision=False`라 GradScaler는 꺼져 있고(`scale()`은 항등), 그냥 `loss.backward()`입니다.
# 역전파 후 두 종류의 그래디언트가 생깁니다:
# - **6개 파라미터의 `.grad`** → G단계에서 Adam이 사용
# - **`info["means2d"].absgrad`** (D6에서 retain) → I단계에서 densification 판단에 사용. 화면공간 2D 위치에 대한 그래디언트의 *절댓값 누적*이라 부호가 상쇄되지 않습니다 (`use_absgrad=True`)
#
# 가우시안 $i$ 의 화면 좌표 $\mu_i^{2D}$ 는 여러 픽셀 $p$ 에 기여하므로 그래디언트는 픽셀별 기여의 합입니다. 일반 grad는 부호가 반대인 기여가 상쇄되어 "양쪽으로 당겨지는" 가우시안(= 쪼개야 할 것)을 놓치지만, absgrad는 픽셀별 절댓값을 더해 그 신호를 보존합니다:
#
# $$
# \frac{\partial \mathcal L}{\partial \mu_i^{2D}} = \sum_{p} \frac{\partial \mathcal L}{\partial \mu_i^{2D}}\Big|_{p},
# \qquad
# \text{absgrad}_i = \sum_{p} \left|\frac{\partial \mathcal L}{\partial \mu_i^{2D}}\Big|_{p}\right|
# \;\ge\; \left|\frac{\partial \mathcal L}{\partial \mu_i^{2D}}\right|
# $$

# %%
loss = torch.stack(list(loss_dict.values())).sum()  # functools.reduce(torch.add, loss_dict.values()) 와 동일
trainer.grad_scaler.scale(loss).backward()

print(f"{'group':14s} {'param shape':>18s} {'|grad| mean':>12s} {'|grad| max':>12s}  lr")
for name, p in model.gauss_params.items():
    g = p.grad
    lr = opt.optimizers[name].param_groups[0]["lr"]
    print(f"{name:14s} {str(tuple(p.shape)):>18s} {g.abs().mean().item():12.3e} {g.abs().max().item():12.3e}  {lr:.2e}")

print("\n관찰: features_rest grad=0 (SH 0차만 사용 중), quats grad≈1e-13 (구는 돌려도 같음) — 두 파라미터는 step 0에서 사실상 학습되지 않음")

g2d = info["means2d"].absgrad  # [1,N,2]
gn = g2d[0].norm(dim=-1)
vis_mask = info["radii"][0] > 0
print(f"\nmeans2d.absgrad {tuple(g2d.shape)} | 보이는 가우시안 중 norm > densify_grad_thresh({model.config.densify_grad_thresh}): {(gn[vis_mask] > model.config.densify_grad_thresh).sum().item():,} / {vis_mask.sum().item():,}")
print("  (안 보이는 가우시안은 grad=0 — 래스터라이저를 안 거쳤으니 당연)")


# %% [markdown]
# ## G. optimizer step — `optimizers.optimizer_scaler_step_some(grad_scaler, needs_step)` ([optimizers.py:159-172](../nerfstudio/engine/optimizers.py#L159-L172))
#
# 그룹마다 별도 Adam 인스턴스가 있고 각자 `step()`을 밟습니다. `max_norm`(grad clipping)은 splatfacto에선 전부 None.
# 첫 step 직후 Adam 내부 상태(`exp_avg`, `exp_avg_sq`)가 파라미터와 같은 shape로 생깁니다 — I단계에서 이 텐서들이 리사이즈되는 걸 볼 겁니다.
#
# Adam 갱신식 ($g_t$ = 그래디언트, `exp_avg` $= m_t$, `exp_avg_sq` $= v_t$, splatfacto 기본 $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-15}$):
#
# $$
# m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
# v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,\qquad
# \theta_t = \theta_{t-1} - \eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon}
# $$
#
# $t=1$ 이면 $\hat m_1 = g_1$, $\hat v_1 = g_1^2$ 이라 갱신량이 $\eta\,\dfrac{g_1}{|g_1|} = \pm\eta$ 로 그래디언트 크기와 무관하게 정확히 lr 만큼 움직입니다 — 아래 출력에서 `|Δparam| ≈ lr` 인 이유. 원소별로 독립이라 가우시안 개수 $N$ 이 바뀌어도 각 행의 $(m, v)$ 만 같은 인덱스로 옮기면 됩니다.

# %%
before = {k: v.detach().clone() for k, v in model.gauss_params.items()}
print("step 전 Adam state 개수:", {k: len(o.state) for k, o in opt.optimizers.items()})

needs_step = [g for g in opt.parameters if STEP % trainer.gradient_accumulation_steps[g] == trainer.gradient_accumulation_steps[g] - 1]
opt.optimizer_scaler_step_some(trainer.grad_scaler, needs_step)

print("step 후 Adam state:", {k: list(o.state[list(o.state)[0]].keys()) for k, o in opt.optimizers.items() if o.state})
print(f"\n{'group':14s} {'|Δparam| mean':>14s} {'|Δparam| max':>14s}   해석")
notes = {
    "means": "lr 1.6e-4 → 첫 스텝 Adam은 |Δ|≈lr (모멘트 정규화로 크기가 lr에 맞춰짐)",
    "scales": "log 공간에서 이동",
    "quats": "grad≈1e-13: 초기 가우시안이 '구'라서 회전이 렌더에 영향 없음 → 타원체가 된 뒤에야 의미 생김",
    "features_dc": "SH 0차 — 색이 가장 빨리 움직임 (lr 2.5e-3)",
    "features_rest": "grad=0: step 0은 sh_degree_to_use=0 → 고차 밴드가 forward에 안 쓰임 (1000스텝 후부터)",
    "opacities": "logit 공간, lr 0.05로 가장 큼",
}
for k, v in model.gauss_params.items():
    dlt = (v.detach() - before[k]).abs()
    print(f"{k:14s} {dlt.mean().item():14.3e} {dlt.max().item():14.3e}   {notes[k]}")


# %% [markdown]
# ## H. scheduler — `optimizers.scheduler_step_all(step)` ([optimizers.py:183-193](../nerfstudio/engine/optimizers.py#L183-L193))
#
# 스케줄러가 있는 그룹은 `means`만 (camera_opt/bilateral_grid는 꺼져 있어 그룹 자체가 없음). 1.6e-4 → 1.6e-6으로 30k 스텝 동안 지수감쇠:
# 초반엔 위치를 크게 움직여 구조를 잡고, 후반엔 위치는 거의 고정하고 색/모양만 다듬습니다.
#
# `ExponentialDecayScheduler`는 로그 공간의 선형 보간, 즉 스텝당 일정 비율로 감쇠합니다 ($\eta_0 = 1.6\times10^{-4},\ \eta_T = 1.6\times10^{-6},\ T = 30000$):
#
# $$
# \eta(t) = \eta_0 \left(\frac{\eta_T}{\eta_0}\right)^{t/T}
# = \eta_0 \exp\!\Big(\frac{t}{T}\ln\frac{\eta_T}{\eta_0}\Big),
# \qquad
# \frac{\eta(t+1)}{\eta(t)} = \Big(\frac{1}{100}\Big)^{1/30000} \approx 0.999847
# $$
# GradScaler의 scale이 줄어든 스텝(= inf/nan으로 optimizer step을 건너뛴 스텝)에는 스케줄러도 건너뜁니다 — mixed_precision이 꺼져 있으면 항상 진행.

# %%
scale_before = trainer.grad_scaler.get_scale()
trainer.grad_scaler.update()
lr_before = opt.schedulers["means"].get_last_lr()[0]
if scale_before <= trainer.grad_scaler.get_scale():
    opt.scheduler_step_all(STEP)
lr_after = opt.schedulers["means"].get_last_lr()[0]
print(f"means lr: {lr_before:.6e} → {lr_after:.6e}  (감쇠비 {lr_after/lr_before:.6f}/step, 30k 스텝이면 ×{(lr_after/lr_before)**30000:.1e} ≈ 1.6e-6/1.6e-4)")

# 전체 스케줄 그려보기
steps = np.arange(0, 30001, 100)
lr_curve = 1.6e-4 * (1.6e-6 / 1.6e-4) ** (steps / 30000)
plt.figure(figsize=(7, 3)), plt.semilogy(steps, lr_curve), plt.axvline(15000, c="r", ls="--", label="stop_split_at")
plt.title("learning_rate/means (ExponentialDecay)"), plt.xlabel("step"), plt.legend(), plt.show()


# %% [markdown]
# ## I. AFTER 콜백 — `strategy.step_post_backward` 를 펼치면 (gsplat `DefaultStrategy`)
#
# ```python
# if step >= refine_stop_iter(15000): return                  # 이후엔 아무것도 안 함
# _update_state(params, state, info)                          # grad2d[i] += |means2d.absgrad[i]|·(W/2, H/2) 스케일, count[i] += 1 (보인 가우시안만)
# if step > 500 and step % 100 == 0 and step % 3000 >= 261:   # refine 조건 (알파 리셋 직후 261스텝은 쉼)
#     _grow_gs   : grad2d/count > 0.0008 인 것 중  scale < 0.01 → 복제,  ≥ 0.01 → 2개로 분할(원본 삭제)
#     _prune_gs  : sigmoid(opacity) < 0.1 삭제,  (step < 4000) 화면공간 반경 > 0.15 삭제
#     state 리셋
# if step % 3000 == 0: reset_opa(value = 0.1·2 = 0.2)       # opacity = min(opacity, 0.2) → step 0에도 호출됨!
# ```
# `_update_state`가 누적하는 `grad2d / count`(= 보였을 때의 평균 화면공간 그래디언트)가 바로 "이 가우시안이 아직 잘 못 맞춘 영역을 덮고 있다"는 신호입니다.
#
# **통계 누적.** absgrad는 NDC 단위($[-1,1]$)라서 $(W/2, H/2)$ 를 곱해 픽셀 단위로 바꾼 뒤, 이번 스텝에 보인($r_i > 0$) 가우시안만 누적합니다:
#
# $$
# \text{grad2d}_i \mathrel{+}= \Big\|\,\text{absgrad}_i \odot \big(\tfrac W2, \tfrac H2\big)\Big\|_2,
# \qquad
# \text{count}_i \mathrel{+}= 1
# \qquad (\text{if } r_i > 0)
# $$
#
# **grow (100스텝마다).** 평균 화면공간 그래디언트가 임계값을 넘는 가우시안을 크기에 따라 두 갈래로 처리합니다 ($\tau_g$ = `densify_grad_thresh` = 0.0008, $\tau_s$ = `densify_size_thresh` = 0.01):
#
# $$
# \bar g_i = \frac{\text{grad2d}_i}{\text{count}_i} > \tau_g
# \;\Longrightarrow\;
# \begin{cases}
# \max_k \exp(\tilde s_{ik}) < \tau_s & \text{복제(clone): 같은 파라미터를 하나 더} \\[2pt]
# \max_k \exp(\tilde s_{ik}) \ge \tau_s & \text{분할(split): } \mu' = \mu + R(q)\,(s \odot z),\; z\sim\mathcal N(0,I),\; s' = s/1.6 \text{ 로 2개}
# \end{cases}
# $$
#
# 작은 가우시안이 큰 그래디언트를 받으면 "표현력이 모자란" 상태(복제해서 채움), 큰 가우시안이면 "너무 뭉뚱그려 덮고 있는" 상태(쪼개서 세분화)라는 해석입니다.
#
# **prune.** $\sigma(\tilde\alpha_i) < 0.1$ 인 것을 지우고, step < 4000 동안은 화면공간 반경이 화면의 15%를 넘는 거대 가우시안도 지웁니다.
#
# **알파 리셋 (3000스텝마다).** 불투명도를 로짓 공간에서 위로 clamp 합니다 — 모든 가우시안을 반투명으로 되돌려, 쓸모없는 것은 다음 prune 전까지 다시 불투명해지지 못하고 걸러지게 합니다:
#
# $$
# \tilde\alpha_i \leftarrow \min\big(\tilde\alpha_i,\ \operatorname{logit}(0.2)\big)
# \quad\Longleftrightarrow\quad
# \sigma(\tilde\alpha_i) \leftarrow \min\big(\sigma(\tilde\alpha_i),\ 0.2\big)
# $$
# grow/prune 시 파라미터 텐서를 새로 만들면서 **optimizer의 param_groups와 exp_avg/exp_avg_sq도 같은 인덱스로 재배열**합니다 — A단계에서 optimizers를 모델에 꽂아준 이유.

# %%
st = model.strategy_state
print("state 전:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in st.items()})
opa_before = torch.sigmoid(model.opacities).detach()

model.step_post_backward(STEP)  # == AFTER 콜백

print("state 후:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in st.items()})
seen = st["count"] > 0
print(f"count>0 (이번 스텝에 보인) 가우시안: {seen.sum().item():,} | grad2d 평균(보인 것만): {(st['grad2d'][seen] / st['count'][seen]).mean().item():.5f}  vs thresh {model.config.densify_grad_thresh}")
opa_after = torch.sigmoid(model.opacities).detach()
print(f"reset_opa(step % 3000 == 0): opacity max {opa_before.max().item():.3f} → {opa_after.max().item():.3f}  (0.2로 clamp; 초기값 0.1이라 이번엔 변화 없음)")
refine_now = STEP > model.config.warmup_length and STEP % model.config.refine_every == 0 and STEP % (model.config.reset_alpha_every * model.config.refine_every) >= model.strategy.pause_refine_after_reset
print(f"이번 스텝 refine 조건: step>{model.config.warmup_length} and step%100==0 and step%3000>={model.strategy.pause_refine_after_reset} → {refine_now}")


# %% [markdown]
# ## A~I 를 함수 하나로 — 그리고 첫 densification 까지 700스텝
#
# 위 단계들을 그대로 묶으면 `train_iteration`과 동일한 함수가 됩니다. 이걸로 step 1~700을 돌리며
# **step 600, 700에서 가우시안이 늘어날 때 Adam 상태가 어떻게 따라 커지는지**를 확인합니다.

# %%
def train_step(step: int):
    """trainer.train() 루프 1회 + train_iteration() 을 A~I 순서로 펼친 것"""
    pipeline.train()
    for cb in trainer.callbacks:                                                     # A
        cb.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
    opt.zero_grad_some([g for g in opt.parameters if step % trainer.gradient_accumulation_steps[g] == 0])  # B
    camera, batch = dm.next_train(step)                                              # C
    outputs = model.get_outputs(camera)                                              # D (D1~D7)
    metrics = model.get_metrics_dict(outputs, batch)                                 # E
    loss_dict = model.get_loss_dict(outputs, batch, metrics)
    loss = torch.stack(list(loss_dict.values())).sum()
    trainer.grad_scaler.scale(loss).backward()                                       # F
    opt.optimizer_scaler_step_some(                                                  # G
        trainer.grad_scaler,
        [g for g in opt.parameters if step % trainer.gradient_accumulation_steps[g] == trainer.gradient_accumulation_steps[g] - 1],
    )
    scale = trainer.grad_scaler.get_scale()
    trainer.grad_scaler.update()
    if scale <= trainer.grad_scaler.get_scale():                                     # H
        opt.scheduler_step_all(step)
    for cb in trainer.callbacks:                                                     # I
        cb.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
    return loss.item(), metrics


def adam_state_shapes():
    out = {}
    for name, o in opt.optimizers.items():
        p = o.param_groups[0]["params"][0]
        s = o.state.get(p, {})
        out[name] = (tuple(p.shape), tuple(s["exp_avg"].shape) if "exp_avg" in s else None)
    return out


hist = {"step": [], "loss": [], "n": []}
for step in range(1, 701):
    if step in (599, 600, 699, 700):
        n_before = model.num_points
        shapes_before = adam_state_shapes()["means"]
    loss_val, metrics = train_step(step)
    hist["step"].append(step), hist["loss"].append(loss_val), hist["n"].append(model.num_points)
    if step in (600, 700):
        print(f"\n[step {step}] 가우시안 {n_before:,} → {model.num_points:,}")
        print(f"   means param / Adam exp_avg shape: {shapes_before} → {adam_state_shapes()['means']}")
        # 새로 생긴 가우시안의 Adam 모멘트는 0으로 초기화되어 들어옴 (gsplat _update_param_with_optimizer)
        o = opt.optimizers["means"]
        p = o.param_groups[0]["params"][0]
        print(f"   param 객체가 교체됨: gauss_params['means'] is param_groups[0]['params'][0] → {model.gauss_params['means'] is p}")
        print(f"   state['grad2d'] 리셋: {tuple(st['grad2d'].shape)}, sum={st['grad2d'].sum().item():.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
ax[0].plot(hist["step"], hist["loss"], lw=0.6), ax[0].set_yscale("log"), ax[0].set_title("loss (step 1~700)")
ax[1].plot(hist["step"], hist["n"]), ax[1].axvline(500, c="gray", ls=":", label="warmup end"), ax[1].set_title("가우시안 개수 — 600, 700에서 refine"), ax[1].legend()
plt.show()


# %% [markdown]
# ## 정리 — `train_iteration` 한 줄에 숨어 있던 것
#
# | 단계 | 무엇이 바뀌나 | 어디서 |
# |---|---|---|
# | A | `model.step`, `model.optimizers` | 콜백 |
# | B | `.grad` = 0 | 6개 Adam |
# | C | 이미지 1장 선택 | 데이터매니저 캐시 |
# | D | 가우시안 → 픽셀 (래스터화) | gsplat CUDA |
# | E | loss 스칼라 1개 | L1 + SSIM |
# | F | 6개 `.grad` + `means2d.absgrad` | autograd |
# | G | **6개 파라미터 값** | Adam ×6 |
# | H | `means` lr | 스케줄러 |
# | I | `grad2d/count` 누적 → (100스텝마다) **파라미터 개수 + Adam 상태 크기** | gsplat strategy |
#
# 파라미터 *값*은 G에서만, 파라미터 *개수*는 I에서만 바뀝니다. 이 둘이 분리되어 있고 I가 optimizer 상태까지 손대야 하기 때문에 A의 주입이 필요합니다.

# %%
del trainer, pipeline, model, dm, opt
torch.cuda.empty_cache()
print("done")
