# Splatfacto `train_iteration()` 한 번 — A~I 아홉 단계

## 한 줄 요약
`Trainer.train()` 루프 한 바퀴(BEFORE 콜백 → `train_iteration()` → AFTER 콜백)를 안쪽까지 펼치면 다음 아홉 단계가 됩니다.

```
A. BEFORE 콜백      model.step_cb(optimizers, step)         step 갱신, optimizers 주입
B. zero_grad        optimizers.zero_grad_some(...)
C. 데이터            datamanager.next_train(step)            (camera, {"image"})
D. forward          model.get_outputs(camera)               카메라 보정 → 해상도 스케줄 → viewmat/K → rasterization → 배경 합성
E. metrics/loss     get_metrics_dict / get_loss_dict        PSNR / 0.8·L1 + 0.2·(1−SSIM)
F. backward         grad_scaler.scale(loss).backward()
G. optimizer step   optimizers.optimizer_scaler_step_some    6개 Adam 각각 step
H. scheduler        optimizers.scheduler_step_all            means lr 지수감쇠
I. AFTER 콜백       model.step_post_backward(step)          통계 누적 → (100스텝마다) 분할/복제/컬링 → (3000마다) 알파 리셋
```

## 암기 힌트
- **A·I는 콜백**으로 샌드위치 구조: 앞(BEFORE)은 *꽂아 주기*, 뒤(AFTER)는 *구조 바꾸기*.
- 가운데 **B~H는 일반적인 PyTorch 학습 루프** 그대로: `zero_grad → 데이터 → forward → loss → backward → step → scheduler`. 이 순서는 어떤 딥러닝 코드에서나 같으니 외울 것은 사실상 A와 I만.
- 코드 위치: A, I는 `trainer.py:246-273`의 `train()` 루프에 있고, B~H는 `trainer.py:487-530`의 `train_iteration()` 안에 있다.

## 단계별 설명

### A. BEFORE 콜백 — `model.step_cb(optimizers, step)`
Trainer는 모델 생성 시 optimizer를 넘기지 않는다. 대신 매 스텝 시작에 `TrainingCallbackLocation.BEFORE_TRAIN_ITERATION` 콜백으로 `step`과 `optimizers`를 모델에 꽂아 준다(`model.optimizers is trainer.optimizers.optimizers` — 같은 객체 공유). I단계에서 densification이 가우시안 개수를 바꾸면 Adam 모멘트 텐서(`exp_avg`, `exp_avg_sq`)도 같은 크기로 늘려야 하므로 모델이 optimizer에 손을 댈 수 있어야 한다.

### B. zero_grad — `optimizers.zero_grad_some(needs_zero)`
gradient accumulation을 위해 그룹별로 "이번 스텝에 0으로 만들 그룹"을 고르지만, splatfacto 기본은 accumulation 1이므로 매 스텝 6개 그룹 전부 0으로 만든다.

### C. 데이터 — `datamanager.next_train(step)`
`FullImageDatamanager`가 캐시된 이미지 중 이번 epoch에 아직 안 본 카메라 하나를 pop 해서 `(Cameras[1], {"image": uint8 [H,W,3]})`를 돌려준다. 다 쓰면 다시 섞는다.

### D. forward — `model.get_outputs(camera)`
안쪽은 다시 D1~D7로 나뉜다: 카메라 보정(`camera_optimizer`, 기본 off) → 해상도 스케줄 `d = 2^max(num_downscales − step//resolution_schedule, 0)` → `viewmat`/`K` 계산(OpenGL→OpenCV 축 반전) → SH 차수 `min(step//1000, 3)` → `gsplat.rasterization(...)` → `strategy.step_pre_backward`(`means2d.retain_grad()`) → 랜덤 배경 합성 `rgb = render + (1−alpha)·b`.

### E. metrics / loss
`get_metrics_dict`가 PSNR과 gaussian_count를, `get_loss_dict`가 `main_loss = 0.8·L1 + 0.2·(1−SSIM)`를 계산한다 (`ssim_lambda = 0.2`). GT는 학습 해상도로 평균 풀링해 축소한다.

### F. backward — `grad_scaler.scale(loss).backward()`
`mixed_precision=False`라 GradScaler는 항등. 역전파 후 두 종류의 그래디언트가 생긴다: 6개 파라미터의 `.grad`(G에서 사용)와 `info["means2d"].absgrad`(I에서 densification 판단에 사용).

### G. optimizer step — `optimizers.optimizer_scaler_step_some(grad_scaler, needs_step)`
`means / scales / quats / features_dc / features_rest / opacities` 6개 그룹마다 별도 Adam이 있고 각자 `step()`을 밟는다. 첫 스텝의 Adam은 `|Δparam| ≈ lr`. **파라미터 값**이 바뀌는 단계.

### H. scheduler — `optimizers.scheduler_step_all(step)`
스케줄러가 있는 그룹은 `means`만: 1.6e-4 → 1.6e-6으로 30k 스텝 지수감쇠. GradScaler의 scale이 줄어든 스텝(inf/nan으로 step을 건너뛴 경우)은 스케줄러도 건너뛴다.

### I. AFTER 콜백 — `model.step_post_backward(step)` → `strategy.step_post_backward`
gsplat `DefaultStrategy`: `means2d.absgrad`를 픽셀 단위로 환산해 `grad2d/count`에 누적 → `step > 500 and step % 100 == 0`이면 grow(복제/분할)·prune(불투명도 < 0.1, 거대 가우시안) → `step % 3000 == 0`이면 opacity를 0.2로 clamp(알파 리셋). 이때 파라미터 텐서를 새로 만들면서 optimizer의 `param_groups`와 `exp_avg/exp_avg_sq`도 같은 인덱스로 재배열한다 — A단계에서 optimizers를 꽂아준 이유. **파라미터 개수(N)**가 바뀌는 단계.

## G vs I 대비
| | G. optimizer step | I. AFTER 콜백 |
|---|---|---|
| 바꾸는 것 | 파라미터 **값** | 파라미터 **개수** (가우시안 N) |
| 주기 | 매 스텝 | 통계는 매 스텝, 구조 변경은 100스텝마다(500 이후), 알파 리셋 3000마다 |
| 사용하는 grad | `param.grad` | `means2d.absgrad` |
| Adam 상태 | 값 갱신 | shape 리사이즈 |

## 출처
- `.fm/assets/splatfacto_train_step.py` — A~I를 셀 하나씩 재현하고 원본과 일치를 검증한 노트북
- `nerfstudio/engine/trainer.py` `train()` L246-273, `train_iteration()` L487-530
- `nerfstudio/models/splatfacto.py` `step_cb` L407-410, `get_outputs` L485-600, loss L631-689
