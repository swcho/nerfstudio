# `cull_alpha_thresh` — 불투명도 기반 가우시안 컬링 임계값

## 한 줄 요약

`cull_alpha_thresh`(기본 **0.1**)는 **불투명도(opacity)가 이 값보다 낮은 가우시안을 제거(컬링)하는 임계값**이다. gsplat의 densification 전략에 그대로 전달되며, **0.005처럼 낮게 잡으면 가우시안이 덜 지워져 품질이 올라간다**. `splatfacto-big`과 `splatfacto-mcmc`가 실제로 0.005를 사용한다.

## 정의 위치

`SplatfactoModelConfig` (splatfacto.py):

```python
cull_alpha_thresh: float = 0.1
"""threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
```

docstring 자체가 이미 품질 팁을 담고 있다 — "더 낮은 값(예: 0.005)으로 설정하면 품질이 높아진다."

## 어디로 전달되는가 — gsplat 전략과의 연결

Splatfacto는 densification/pruning 로직을 직접 구현하지 않고 gsplat 라이브러리의 전략 객체에 위임한다. `populate_modules()`에서 이 값이 전략별로 다른 이름의 인자로 들어간다.

**1) `strategy == "default"` → `DefaultStrategy`의 `prune_opa`:**

```python
self.strategy = DefaultStrategy(
    prune_opa=self.config.cull_alpha_thresh,   # ← 여기
    grow_grad2d=self.config.densify_grad_thresh,
    prune_scale3d=self.config.cull_scale_thresh,
    prune_scale2d=self.config.cull_screen_size,
    ...
)
```

**2) `strategy == "mcmc"` → `MCMCStrategy`의 `min_opacity`:**

```python
self.strategy = MCMCStrategy(
    cap_max=self.config.max_gs_num,
    min_opacity=self.config.cull_alpha_thresh,  # ← 여기
    ...
)
```

즉 같은 config 값 하나가 default 전략에서는 "이 불투명도 미만이면 prune", MCMC 전략에서는 "죽은(dead) 가우시안으로 판정해 재배치(relocation) 대상으로 삼는 최소 불투명도"로 쓰인다.

## 왜 필요한가

- 학습 중 opacity가 거의 0에 수렴한 가우시안은 렌더링에 기여하지 못하면서 메모리와 연산만 차지한다. `refine_every`(기본 100 스텝) 주기마다 이런 가우시안을 걷어내 모델을 가볍게 유지한다.
- 3DGS 계열에는 주기적 **opacity reset**(`reset_alpha_every`)이 있어서, 리셋 후 다시 불투명도를 회복하지 못한 가우시안들이 이 임계값에 걸려 정리된다. 두 메커니즘이 짝을 이뤄 반투명 찌꺼기(floater)를 억제한다.

## 품질 vs 효율 트레이드오프

| 값 | 효과 |
|---|---|
| 0.1 (기본, `splatfacto`) | 공격적으로 컬링 → 가우시안 수가 적어 빠르고 메모리 절약, 품질은 다소 손해 |
| 0.005 (`splatfacto-big`, `splatfacto-mcmc`) | 반투명 가우시안까지 살려둠 → 미세한 디테일·부드러운 경계 표현이 좋아짐, 대신 가우시안 수가 크게 늘어 VRAM/속도 비용 증가 |

`method_configs.py`에서 확인:

- `splatfacto-big`: `cull_alpha_thresh=0.005` (+ `densify_grad_thresh=0.0005`로 densify도 더 공격적)
- `splatfacto-mcmc`: `strategy="mcmc"`, `cull_alpha_thresh=0.005` (+ `stop_split_at=25000`)

참고로 원조 3DGS 논문(Kerbl et al. 2023)의 pruning 임계값은 0.005였다 — 즉 "big" 변형이 원논문 설정에 더 가깝고, 기본 `splatfacto`는 효율을 위해 임계값을 20배 높인(0.1) 셈이다.

## 암기 포인트

- 이름 매핑: `cull_alpha_thresh` → DefaultStrategy `prune_opa` / MCMCStrategy `min_opacity`
- 숫자: 기본 **0.1**, 고품질 프리셋(big/mcmc)은 **0.005** (20배 차이)
- 방향: **낮출수록** 덜 지워짐 → 품질 ↑, 가우시안 수·메모리 ↑
