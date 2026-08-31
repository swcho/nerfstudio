# `cull_scale_thresh` vs `cull_screen_size` — 두 가지 "거대 가우시안" 컬링 기준

## 한 줄 요약

둘 다 **너무 커진 가우시안을 제거(prune)** 하는 임계값이지만, 크기를 재는 공간이 다르다.

| 설정 | 기본값 | 측정 공간 | gsplat `DefaultStrategy` 인자 | 의미 |
|---|---|---|---|---|
| `cull_scale_thresh` | **0.5** | **월드(3D) 공간** | `prune_scale3d` | 3D 스케일이 임계값(× scene scale)을 넘는 거대 가우시안 제거 |
| `cull_screen_size` | **0.15** | **화면(2D) 공간** | `prune_scale2d` | 렌더링 시 화면의 **15% 이상**을 차지하는 가우시안 제거 |

## 설정 정의 (`SplatfactoModelConfig`)

```python
cull_scale_thresh: float = 0.5
"""threshold of scale for culling huge gaussians"""
...
cull_screen_size: float = 0.15
"""if a gaussian is more than this percent of screen space, cull it"""
```

## gsplat 전략으로 전달되는 방식

Splatfacto는 컬링/증식 로직을 직접 구현하지 않고 gsplat의 `DefaultStrategy`에 위임한다. 이때 nerfstudio 설정 이름이 gsplat 인자 이름으로 매핑된다.

```python
self.strategy = DefaultStrategy(
    prune_opa=self.config.cull_alpha_thresh,        # 불투명도 컬링 (0.1)
    ...
    prune_scale3d=self.config.cull_scale_thresh,    # 3D 스케일 컬링 (0.5)
    prune_scale2d=self.config.cull_screen_size,     # 2D 화면 크기 컬링 (0.15)
    refine_scale2d_stop_iter=self.config.stop_screen_size_at,  # 4000
    ...
)
```

즉 암기 포인트: **`cull_scale_thresh` → `prune_scale3d`(월드), `cull_screen_size` → `prune_scale2d`(화면)**.

## 왜 두 기준이 따로 필요한가

- **3D 스케일 컬링(`prune_scale3d`)**: densification 과정에서 스케일이 폭주해 장면 전체를 덮는 "안개 같은" 거대 가우시안이 생길 수 있다. 월드 좌표에서 가우시안의 최대 축 스케일이 `0.5 × scene_scale`을 넘으면 제거한다. 카메라와 무관한 절대적 기준.
- **2D 화면 크기 컬링(`prune_scale2d`)**: 3D에서는 크지 않아도 카메라에 가까이 있으면 투영된 반경이 화면을 크게 덮어 렌더링 아티팩트(팝핑, 큰 블롭)를 만든다. 투영 반경이 화면 크기의 15%를 넘으면 제거한다. 원조 3DGS 논문의 "view-space에서 너무 큰 가우시안 제거"에 해당하는 기준.

## 함께 알아두면 좋은 관련 설정

- `cull_alpha_thresh = 0.1` (`prune_opa`): 불투명도가 임계값 미만인 **투명한** 가우시안을 컬링하는 세 번째 기준. 즉 컬링 조건은 "너무 투명하거나(α), 3D로 너무 크거나, 화면에서 너무 크거나"의 OR이다.
- `split_screen_size = 0.05` (`grow_scale2d`): 화면의 5%를 넘으면 **분할**(split) 대상 — 15%를 넘으면 아예 제거. 같은 화면 공간 기준이지만 하나는 성장(refine), 하나는 제거.
- `stop_screen_size_at = 4000` (`refine_scale2d_stop_iter`): 화면 크기 기반 컬링/분할은 **step 4000 이후 중단**된다. 반면 3D 스케일 컬링은 refinement가 지속되는 동안(`stop_split_at`까지) 계속 적용된다.
- gsplat `DefaultStrategy` 내부에서는 3D 스케일 컬링이 첫 opacity reset 주기(`reset_every`) 이후부터 적용된다 — 초기 워밍업 단계의 큰 가우시안을 성급하게 지우지 않기 위함이다.
- 참고: `strategy="mcmc"`(MCMCStrategy)를 쓰면 이 두 설정은 전달되지 않는다 — MCMC는 `min_opacity`(=`cull_alpha_thresh`) 기반 재배치만 사용한다.

## 기억법

- **scale** = 3D 세계의 자(scale) → 월드 공간, 0.**5** ("절반짜리 씬 크기면 너무 크다").
- **screen** = 모니터 화면 → 화면 공간 비율, 0.**15** ("화면의 15%를 덮으면 너무 크다").
