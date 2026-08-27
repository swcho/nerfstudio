# I단계 — gsplat `DefaultStrategy.step_post_backward`의 전체 흐름

## 한 줄 답
`step ≥ refine_stop_iter(15000)`이면 **즉시 return**. 아니면 `_update_state`로 `grad2d`/`count`를 누적하고, **`step > 500` 이고 `step % 100 == 0` 이고 `step % 3000 ≥ 261`** 이면 `_grow_gs`(복제/분할) → `_prune_gs` → state 리셋을 수행. 마지막으로 **`step % 3000 == 0`** 이면 `reset_opa`(opacity를 0.2로 clamp).

## 어디서 호출되나
`Trainer.train_iteration()`의 마지막 단계(AFTER 콜백)에서 `model.step_post_backward(step)`이 호출되고, splatfacto는 이를 gsplat 전략 객체로 그대로 위임합니다 (`nerfstudio/models/splatfacto.py:365-374`):

```python
self.strategy.step_post_backward(params=self.gauss_params, optimizers=self.optimizers,
                                 state=self.strategy_state, step=self.step, info=self.info, packed=False)
```

`info`는 D단계 `rasterization()`이 돌려준 dict(`means2d`, `radii`, `width`, `height`, `n_cameras` …)이고, D6에서 `step_pre_backward`가 `means2d.retain_grad()`를 걸어 두었기 때문에 backward 후 `means2d.absgrad`가 살아 있습니다.

## gsplat 원본 (gsplat 1.4.0, `strategy/default.py`)

```python
def step_post_backward(self, params, optimizers, state, step, info, packed=False):
    if step >= self.refine_stop_iter:            # ① 15000 이후 → 아무것도 안 함
        return
    self._update_state(params, state, info, packed)   # ② 통계 누적 (매 스텝)
    if (step > self.refine_start_iter             # ③ refine 게이트
        and step % self.refine_every == 0
        and step % self.reset_every >= self.pause_refine_after_reset):
        n_dupli, n_split = self._grow_gs(params, optimizers, state, step)   # 복제/분할
        n_prune = self._prune_gs(params, optimizers, state, step)          # 컬링
        state["grad2d"].zero_(); state["count"].zero_()                    # 통계 리셋
        if self.refine_scale2d_stop_iter > 0: state["radii"].zero_()
        torch.cuda.empty_cache()
    if step % self.reset_every == 0:              # ④ 알파 리셋
        reset_opa(params, optimizers, state, value=self.prune_opa * 2.0)
```

주의할 점: ③과 ④는 `elif`가 아니라 **독립된 두 `if`** 입니다. 다만 ③은 `step % 3000 ≥ 261`을 요구하므로 `step % 3000 == 0`인 스텝에서는 절대 refine이 돌지 않아, 실질적으로 한 스텝에서 둘이 같이 발화하는 일은 없습니다.

## nerfstudio 설정 → DefaultStrategy 인자 매핑 (`splatfacto.py:264-279`)

| DefaultStrategy 인자 | nerfstudio config | 기본값 | 의미 |
|---|---|---|---|
| `refine_stop_iter` | `stop_split_at` | 15000 | ① 이후 전략 완전 정지 |
| `refine_start_iter` | `warmup_length` | 500 | `step > 500`부터 refine |
| `refine_every` | `refine_every` | 100 | 100스텝마다 refine |
| `reset_every` | `reset_alpha_every × refine_every` | 30 × 100 = **3000** | 알파 리셋 주기 |
| `pause_refine_after_reset` | `num_train_data + refine_every` | 161 + 100 = **261** (이 데이터셋) | 리셋 후 쉬는 구간 |
| `prune_opa` | `cull_alpha_thresh` | 0.1 | opacity < 0.1 컬링; 리셋값 = 2 × 0.1 = 0.2 |
| `grow_grad2d` | `densify_grad_thresh` | 0.0008 | 평균 화면공간 grad 임계값 |
| `grow_scale3d` | `densify_size_thresh` | 0.01 | 이하 → 복제, 초과 → 분할 |
| `grow_scale2d` | `split_screen_size` | 0.05 | 화면 반경 기준 분할 (step < 4000) |
| `prune_scale3d` | `cull_scale_thresh` | 0.5 | 너무 큰 3D 가우시안 컬링 (step > 3000) |
| `prune_scale2d` | `cull_screen_size` | 0.15 | 화면 15% 넘는 가우시안 컬링 |
| `refine_scale2d_stop_iter` | `stop_screen_size_at` | 4000 | 화면 크기 기준 grow/prune 종료 |
| `absgrad` | `use_absgrad` | True | `means2d.absgrad` 사용 (AbsGS) |

`pause_refine_after_reset`이 261인 이유: 알파 리셋 직후엔 모든 가우시안이 반투명이라 그래디언트 통계가 왜곡됩니다. **학습 이미지 수(161) + 1주기(100)** 만큼 기다려 모든 뷰를 최소 한 번씩 다시 본 뒤에 refine을 재개하게 합니다. 즉 3000, 3100, 3200에서는 refine이 쉬고 3300부터 재개.

## 각 단계 상세

### ② `_update_state` — 매 스텝 누적
- `absgrad`는 NDC 단위(`[-1,1]`)이므로 `(W/2·n_cameras, H/2·n_cameras)`를 곱해 픽셀 단위로 변환.
- `radii > 0`(이번 스텝에 화면에 보인) 가우시안만 골라 `grad2d[i] += ‖grad_i‖₂`, `count[i] += 1` (`index_add_`).
- `refine_scale2d_stop_iter > 0`이므로 `state["radii"]`에 정규화된 화면 반경의 최댓값도 유지.
- 첫 호출(step 0)에서 `state["grad2d"]`, `state["count"]`가 `None` → `zeros(N)`으로 lazy 초기화.

### ③-a `_grow_gs` — 복제/분할
- `ḡ = grad2d / max(count, 1)` 가 `0.0008`을 넘는 가우시안 중
  - `max(exp(scales)) ≤ 0.01·scene_scale` → **duplicate** (같은 파라미터 하나 더)
  - 그 외 → **split** (원본 삭제, `s/1.6` 크기의 2개를 `μ + R(q)(s⊙z)`에 생성)
  - step < 4000 이면 `state["radii"] > 0.05`인 것도 split 대상에 추가
- 복제 먼저, 그 다음 분할. 복제로 새로 생긴 것은 이번 분할에서 제외(`is_split`에 `zeros(n_dupli)`를 이어붙임).

### ③-b `_prune_gs` — 컬링
- 항상: `sigmoid(opacity) < 0.1` 삭제.
- **`step > 3000`일 때만** 추가로: `max(exp(scales)) > 0.5·scene_scale` 삭제, 그리고 `step < 4000`이면 `state["radii"] > 0.15` 삭제.
  - 따라서 화면 크기 컬링은 실제로는 **3300 ~ 3900 구간에서만** 동작합니다(3000 초과 & 4000 미만 & refine 스텝).

### ③-c state 리셋
grow/prune 뒤 `grad2d`, `count`, `radii`를 0으로 → 다음 100스텝 통계는 새 가우시안 집합 기준으로 다시 쌓임.

### ④ `reset_opa` — 알파 리셋
`opacities ← min(opacities, logit(0.2))` (로짓 공간 clamp) 후 opacity optimizer의 Adam 상태(`exp_avg`, `exp_avg_sq`)를 0으로 초기화. **step 0에서도 발화**하지만 초기 opacity가 0.1이라 값 변화는 없음(노트북 I단계 셀 출력: `opacity max 0.100 → 0.100`).

## 왜 optimizer가 필요한가
grow/prune/reset은 파라미터 텐서를 **새 `nn.Parameter`로 교체**합니다. gsplat의 `_update_param_with_optimizer`가 optimizer의 `param_groups[0]["params"][0]`와 `exp_avg`/`exp_avg_sq`를 같은 인덱스로 재배열(새 가우시안은 0으로) 해주기 때문에 A단계에서 `model.step_cb`로 optimizers를 모델에 꽂아 둔 것입니다. 노트북의 700스텝 실험은 step 600, 700에서 `means` 파라미터와 Adam `exp_avg` shape가 함께 커지고 `gauss_params['means'] is param_groups[0]['params'][0]`가 유지되며 `grad2d.sum()`이 0으로 리셋됨을 보여 줍니다.

## 스텝 타임라인 — 어느 분기가 발화하나 (num_train_data = 161)

| step | ① return | ② `_update_state` | ③ refine (grow+prune+reset stats) | ④ `reset_opa` | 비고 |
|---|---|---|---|---|---|
| 0 | – | O (state lazy init) | X (`step > 500` 아님) | **O** (`0 % 3000 == 0`) | 초기 opacity 0.1이라 실질 변화 없음 |
| 100 | – | O | X (warmup) | X | |
| 500 | – | O | X (`>` 이므로 500은 제외) | X | warmup 마지막 |
| **600** | – | O | **O** (`600 % 3000 = 600 ≥ 261`) | X | 첫 densification (노트북 실험) |
| 700 | – | O | **O** | X | 두 번째 refine |
| 2900 | – | O | **O** | X | 리셋 전 마지막 refine |
| **3000** | – | O | X (`3000 % 3000 = 0 < 261`) | **O** | 알파 리셋 → opacity ≤ 0.2, Adam 상태 0 |
| 3100 | – | O | X (`100 < 261`, pause) | X | 161장 다시 보는 중 |
| 3200 | – | O | X (`200 < 261`, pause) | X | |
| **3300** | – | O | **O** (`300 ≥ 261`) | X | 재개; `step > 3000`이라 scale3d/screen-size prune도 켜짐 |
| 3900 | – | O | **O** | X | 화면 크기 기준 grow/prune 마지막 (< 4000) |
| 4000 | – | O | **O** (`4000 % 3000 = 1000 ≥ 261`) | X | 이 스텝부터 `radii` 기준 split/prune은 빠짐 (`step < 4000` 거짓) |
| 6000, 9000, 12000 | – | O | X | **O** | 리셋 반복 |
| 14900 | – | O | **O** | X | 마지막 refine |
| **15000** | **O** | X | X | X (`15000 % 3000 == 0`이지만 return이 먼저) | 이후 가우시안 개수 고정 |
| 20000 | O | X | X | X | |

정리하면 refine은 **600 ~ 14900 사이의 100의 배수 중 `step % 3000 ∈ {0,100,200}`이 아닌 것** 에서, 알파 리셋은 **0, 3000, 6000, 9000, 12000** 에서 발화하고, 15000부터는 함수가 아무 일도 하지 않습니다.
