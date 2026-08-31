# `reset_alpha_every` — 주기적 opacity(alpha) 리셋

## 카드 요약

**Q.** `reset_alpha_every`(기본 30)는 무엇을 하는가?

**A.** refinement 30회마다(즉 `reset_alpha_every * refine_every` = 30 × 100 = **3000스텝마다**) 모든 가우시안의 불투명도(alpha)를 리셋한다. 원조 3DGS 논문의 opacity reset 기법으로, 불필요하게 불투명해진 가우시안을 정리하고 floaters를 줄인다.

---

## 1. 코드에서의 위치

### config 정의 (`splatfacto.py` 103행)

```python
reset_alpha_every: int = 30
"""Every this many refinement steps, reset the alpha"""
```

단위가 **스텝이 아니라 "refinement 횟수"** 라는 점이 핵심이다. refinement는 `refine_every`(기본 100) 스텝마다 한 번씩 일어나므로, 실제 리셋 주기는 스텝 단위로 환산해야 한다.

### gsplat `DefaultStrategy`로 전달 (261–281행)

```python
self.strategy = DefaultStrategy(
    prune_opa=self.config.cull_alpha_thresh,          # 0.1
    ...
    reset_every=self.config.reset_alpha_every * self.config.refine_every,  # 30 * 100 = 3000
    refine_every=self.config.refine_every,             # 100
    pause_refine_after_reset=self.num_train_data + self.config.refine_every,
    ...
)
```

즉 nerfstudio의 `reset_alpha_every`(횟수)는 gsplat의 `reset_every`(스텝)로 변환되어 전달된다:

> `reset_every = reset_alpha_every × refine_every = 30 × 100 = 3000 스텝`

이 3000이라는 값은 원조 3DGS(Kerbl et al. 2023) 논문의 "every 3000 iterations opacity reset"과 정확히 일치한다.

---

## 2. 왜 opacity를 주기적으로 리셋하는가 (3DGS 논문 배경)

원조 3DGS 논문(§5.2 Optimization)에서 도입된 기법으로, 두 가지 문제를 동시에 해결한다.

### (1) 과도하게 불투명해진 가우시안 — gradient 차단

알파 블렌딩 렌더링에서 앞쪽 가우시안의 opacity가 1에 가까워지면, 그 **뒤에 있는 가우시안들에는 gradient가 거의 흐르지 않는다** (투과율 T가 0에 수렴). 한번 불투명해진 가우시안은 "자리를 선점"한 채 뒤쪽 구조의 개선을 막아버린다. opacity를 주기적으로 낮게 리셋하면 모든 가우시안이 다시 경쟁하게 되어, 정말 필요한 가우시안만 opacity를 다시 회복한다.

### (2) 카메라 근처의 floaters

최적화 과정에서 카메라에 가까운 위치에 떠 있는 반투명 덩어리(floaters)가 생기기 쉽다. 이들은 소수의 시점에서만 손실을 줄여주는 "치트"이므로, opacity를 리셋하면 다수 시점의 일관된 supervision을 받지 못해 opacity를 회복하지 못하고 → 다음 culling 단계에서 **opacity threshold(`cull_alpha_thresh`=0.1) 미달로 제거**된다.

### (3) culling과의 상호작용 — 사실상 "가지치기 트리거"

리셋 자체가 정리 메커니즘의 절반이다:

1. 3000스텝마다 모든 가우시안의 opacity를 낮은 값으로 떨어뜨림
2. 이후 학습에서 씬 재구성에 실제로 기여하는 가우시안만 gradient로 opacity를 다시 키움
3. 회복하지 못한 가우시안(중복, floaters, 유령)은 refinement 시 `prune_opa` 기준으로 culling

결과적으로 가우시안 수 폭증을 억제하고 메모리/속도를 관리하는 효과도 있다.

---

## 3. gsplat `DefaultStrategy`의 구현 방식

gsplat의 `DefaultStrategy.step_post_backward()`는 `step % reset_every == 0`일 때 `_reset_opa`류 함수를 호출하여, 모든 가우시안의 opacity를 **`prune_opa * 2.0`(= 0.2)를 상한으로 클램프**한다:

```python
# gsplat 내부 (개념적으로)
opacities = torch.clamp(opacities, max=prune_opa * 2.0)  # logit 공간에서 수행
```

- 원조 3DGS 구현은 opacity를 0.01 근처의 고정 소값으로 리셋했지만, gsplat은 "culling 임계값의 2배" 상한으로 클램프한다. 어느 쪽이든 의도는 같다 — **culling 임계값 바로 위**까지 내려놓아, 회복하지 못하면 곧 제거되도록 만드는 것.
- opacity 파라미터는 sigmoid 이전의 logit으로 저장되므로 실제 연산은 `inverse_sigmoid(0.2)`로의 클램프이며, 해당 파라미터의 optimizer 상태(Adam의 exp_avg 등)도 함께 0으로 리셋된다 — 그렇지 않으면 관성(momentum) 때문에 리셋 직후 opacity가 예전 값으로 튕겨 돌아간다.

---

## 4. `pause_refine_after_reset` — 리셋 직후의 유예 기간

276행:

```python
pause_refine_after_reset=self.num_train_data + self.config.refine_every,
```

리셋 직후에는 **모든** 가우시안의 opacity가 0.2 이하다. 이 상태에서 곧바로 `prune_opa=0.1` 기준 culling을 돌리면 멀쩡한 가우시안까지 대량 학살될 수 있다. 그래서 gsplat은 리셋 후 `pause_refine_after_reset` 스텝 동안 refinement(grow/prune)를 일시 중지한다.

값이 `num_train_data + refine_every`인 이유:

- `num_train_data` = 학습 이미지 수. 스텝마다 이미지 하나를 쓰므로, 이만큼 기다리면 **모든 학습 뷰가 최소 한 번씩 렌더링**되어 각 가우시안이 opacity를 회복할 공정한 기회를 얻는다.
- `+ refine_every`는 여유분으로, 다음 refinement 경계와 겹치는 것을 방지한다.

즉 전체 흐름은:

```
... → [3000스텝: opacity 리셋] → (num_train_data + 100 스텝 동안 refinement 정지,
      가우시안들이 opacity 회복 경쟁) → [refinement 재개: 회복 실패한 것들 culling] → ...
```

---

## 5. 관련 파라미터 한눈에

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `refine_every` | 100 | refinement(densify/cull) 주기 (스텝) |
| `reset_alpha_every` | 30 | 이 횟수의 refinement마다 opacity 리셋 → 30×100 = 3000스텝 |
| `cull_alpha_thresh` (`prune_opa`) | 0.1 | 이 opacity 미만이면 culling; 리셋 시 상한은 이 값의 2배(0.2) |
| `pause_refine_after_reset` | `num_train_data + 100` | 리셋 후 refinement 유예 스텝 수 |
| `stop_split_at` (`refine_stop_iter`) | 15000 | 이 스텝 이후 refinement 종료 → 리셋도 더 이상 의미 없음 |

참고로 `strategy="mcmc"`(MCMCStrategy) 경로에는 opacity reset이 없다 — MCMC 전략은 리셋 대신 확률적 relocation과 노이즈 주입으로 같은 문제를 다룬다.

---

## 6. 기억 포인트

- **단위 함정**: `reset_alpha_every=30`은 30스텝이 아니라 refinement 30회 = **3000스텝**.
- **목적**: over-opaque 가우시안이 뒤쪽 gradient를 막는 것 방지 + 카메라 근처 floaters 제거.
- **메커니즘**: opacity를 culling 임계값 근처로 떨어뜨리고 → 회복 경쟁 → 실패자는 culling. 리셋과 culling은 한 세트.
- **안전장치**: `pause_refine_after_reset`으로 모든 뷰가 한 바퀴 돌 때까지 culling을 유예.
