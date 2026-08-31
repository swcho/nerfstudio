# DefaultStrategy의 `pause_refine_after_reset` 설정과 그 이유

## 질문
DefaultStrategy 생성 시 `pause_refine_after_reset`은 어떻게 설정되며 왜 필요한가?

## 핵심 답
`self.num_train_data + self.config.refine_every`로 설정된다. opacity reset 직후에는 모든 가우시안이 거의 투명해져서 gradient 통계가 무의미하므로, **모든 학습 이미지를 최소 한 번씩 다시 볼 때까지** densify/prune(refinement)을 일시 중지하기 위한 값이다.

## 코드 위치 (splatfacto.py, `populate_modules` 내부)

```python
self.strategy = DefaultStrategy(
    prune_opa=self.config.cull_alpha_thresh,          # 0.1
    grow_grad2d=self.config.densify_grad_thresh,      # 0.0008
    ...
    reset_every=self.config.reset_alpha_every * self.config.refine_every,  # 30 * 100 = 3000
    refine_every=self.config.refine_every,            # 100
    pause_refine_after_reset=self.num_train_data + self.config.refine_every,
    absgrad=self.config.use_absgrad,
    ...
)
```

## 배경: opacity reset이란?

3DGS(및 gsplat의 `DefaultStrategy`)는 학습 중 주기적으로(여기서는 `reset_every = reset_alpha_every × refine_every = 30 × 100 = 3000` step마다) 모든 가우시안의 opacity를 아주 낮은 값(대략 culling 임계값 이하, 예: `min(0.01, 2×prune_opa)` 수준)으로 강제 리셋한다.

리셋의 목적:
- 학습 도중 카메라 근처에 쌓인 **floater**(떠다니는 잡음 가우시안) 제거 — 리셋 후 실제로 필요한 가우시안만 optimizer가 다시 opacity를 키워 살아남고, 불필요한 것은 투명한 채로 남아 다음 culling 때 제거된다.
- opacity가 1 근처에서 포화되어 gradient가 죽는 문제 완화.

## 왜 refinement를 일시 중지해야 하나?

`DefaultStrategy`의 densification(grow/split/duplicate)과 pruning은 각 refinement 주기 동안 누적한 **2D 화면공간 positional gradient 통계**(`grow_grad2d` 임계값과 비교)와 **opacity 값**(`prune_opa`와 비교)에 의존한다.

opacity reset 직후에는:
1. 모든 가우시안이 거의 투명 → 렌더링 결과가 정상 이미지와 크게 다름 → 이때 흘러나오는 gradient는 "장면을 개선하는 신호"가 아니라 "투명해진 것을 복구하는 신호"라 densification 판단 근거로 부적절하다.
2. 모든 가우시안의 opacity가 `prune_opa`(0.1)보다 낮음 → 이 상태로 pruning을 돌리면 **멀쩡한 가우시안까지 전부 잘려나갈** 수 있다.

그래서 리셋 후 `pause_refine_after_reset` step 동안은 grow/prune를 건너뛰고, optimizer가 opacity를 정상 수준으로 회복시킬 시간을 준다.

## 왜 하필 `num_train_data + refine_every`인가?

- **`num_train_data`**: splatfacto는 한 step에 학습 이미지 1장을 사용하므로, `num_train_data` step이면 모든 학습 뷰를 대략 한 번씩(1 epoch) 다시 본 셈이다. 모든 뷰에서 photometric loss를 받아야 각 가우시안의 opacity가 장면 전체 기준으로 의미 있게 회복된다. (일부 뷰만 본 상태에서는 아직 안 보인 영역의 가우시안이 투명한 채 남아 있어 통계가 편향된다.)
- **`+ refine_every`**: refinement는 `refine_every`(100) step 간격으로만 실행되므로, 경계에서 pause가 refinement step 직전에 애매하게 끝나 "회복이 덜 된 상태의 통계"로 한 번 refinement가 실행되는 일을 막는 여유분(margin)이다. 즉 최소 한 번의 온전한 refinement 주기만큼 추가로 기다려, 재개 후 첫 densify/prune가 완전히 회복된 상태의 gradient 누적치로 수행되도록 보장한다.

## 정리 표

| 파라미터 | 값 (기본 설정) | 의미 |
|---|---|---|
| `refine_every` | 100 | densify/prune 실행 주기 |
| `reset_alpha_every` | 30 | refinement 30회마다 opacity 리셋 |
| `reset_every` | 30×100 = 3000 | opacity 리셋 주기(step) |
| `pause_refine_after_reset` | `num_train_data + 100` | 리셋 후 refinement 중지 기간(step) |

## 기억 포인트
- 값: **`num_train_data + refine_every`**
- 이유: 리셋 직후엔 전부 투명 → gradient/opacity 통계가 무의미 → **모든 학습 이미지를 1회씩 다시 볼 때까지**(+ 한 주기 여유) refinement 중지.
