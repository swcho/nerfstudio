# `densify_grad_thresh` — densify 대상을 고르는 positional gradient 임계값

## 질문
`densify_grad_thresh`(기본 0.0008)는 무엇을 제어하는가?

## 핵심 답
**positional gradient(화면 공간에서의 위치 기울기)의 norm이 이 임계값을 넘는 가우시안을 densify(분할 split 또는 복제 duplicate) 대상으로 선정**하는 기준값이다. gsplat `DefaultStrategy`의 `grow_grad2d` 인자로 그대로 전달되며, 값을 **낮출수록 더 많은 가우시안이 임계값을 넘으므로 densify가 공격적**이 된다. 품질 지향 프리셋인 `splatfacto-big`은 이 값을 0.0005로 낮춰 쓴다.

## 코드에서 확인하기

### 1) config 정의 (`SplatfactoModelConfig`)

```python
densify_grad_thresh: float = 0.0008
"""threshold of positional gradient norm for densifying gaussians"""
```

docstring 그대로 "가우시안을 densify하기 위한 positional gradient norm의 임계값"이다.

### 2) gsplat 전략으로 전달 (`populate_modules`)

```python
self.strategy = DefaultStrategy(
    prune_opa=self.config.cull_alpha_thresh,
    grow_grad2d=self.config.densify_grad_thresh,   # ← 여기
    grow_scale3d=self.config.densify_size_thresh,
    ...
    absgrad=self.config.use_absgrad,
    ...
)
```

이름이 말해주듯 `grow_grad2d` = "2D(화면 공간) gradient가 크면 grow(개수를 늘려라)". Splatfacto 자체는 densify 로직을 직접 구현하지 않고 gsplat의 `DefaultStrategy`에 위임하며, 이 config 값은 그 전략의 파라미터로 매핑될 뿐이다.

## 왜 positional gradient인가?

3DGS 원 논문(Kerbl et al. 2023)의 adaptive density control 아이디어다.

- 어떤 가우시안의 **화면 투영 위치(2D means)에 대한 loss gradient가 크다** = 최적화가 그 가우시안을 계속 이리저리 옮기고 싶어 한다 = **그 지역이 현재 가우시안 수로는 표현이 부족하다**(under-reconstruction 또는 over-reconstruction)는 신호.
- 그래서 `refine_every`(100 step)마다 누적된 gradient norm 평균이 임계값을 넘는 가우시안을 골라:
  - 크기가 `densify_size_thresh`(0.01)보다 **작으면 복제(duplicate)** — 가우시안이 더 필요한 곳에 개수를 늘림
  - **크면 분할(split)** — 큰 가우시안을 `n_split_samples`(2)개의 작은 것으로 쪼갬

즉 `densify_grad_thresh`는 "누가 densify 후보인가"를 정하고, `densify_size_thresh`는 "후보를 복제할지 분할할지"를 정하는 역할 분담이다.

## 함께 작동하는 파라미터

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `densify_grad_thresh` | 0.0008 | gradient norm이 이 값을 넘으면 densify 후보 (`grow_grad2d`) |
| `use_absgrad` | True | gradient 누적 시 절댓값 gradient(absgrad) 사용 — 부호 상쇄를 막아 신호가 선명해짐 |
| `densify_size_thresh` | 0.01 | 이 크기 미만이면 duplicate, 이상이면 split (`grow_scale3d`) |
| `refine_every` | 100 | densify/cull을 수행하는 주기 |
| `warmup_length` | 500 | 이 step 전에는 refinement(densify 포함) 미수행 |
| `stop_split_at` | 15000 | 이 step 이후 densify 중단 |

## 값을 바꾸면?

- **낮추면(예: 0.0005)**: 더 작은 gradient에도 densify가 발동 → 가우시안 수가 빠르게 늘어 디테일·품질 향상, 대신 메모리와 학습/렌더링 비용 증가.
- **높이면**: densify가 보수적이 되어 가우시안 수가 적게 유지 → 가볍지만 디테일 손실 가능.

실제로 `nerfstudio/configs/method_configs.py`의 `splatfacto-big` 프리셋은 `densify_grad_thresh=0.0005`로 낮춰(아울러 `cull_alpha_thresh=0.005` 등과 함께) 더 많은 가우시안을 허용하는 고품질 구성을 만든다.

## 기억 포인트
- `densify_grad_thresh` = **"positional gradient norm이 크게 흔들리는 가우시안 → 표현력 부족 → 쪼개거나 복제하라"**의 문턱값.
- gsplat `DefaultStrategy`의 **`grow_grad2d`**로 전달된다 (이름 매핑 암기: densify_**grad**_thresh → grow_**grad2d**).
- 낮을수록 공격적: 기본 0.0008, splatfacto-big 0.0005.
