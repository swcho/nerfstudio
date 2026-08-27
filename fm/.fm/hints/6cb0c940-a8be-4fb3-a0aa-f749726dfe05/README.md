# `_update_state`가 누적하는 통계와 그 의미

## 한 줄 요약

gsplat `DefaultStrategy._update_state`는 매 스텝, **이번 렌더에 실제로 보인 가우시안**($r_i > 0$)에 대해서만 두 가지 러닝 통계를 누적한다.

$$
\text{grad2d}_i \mathrel{+}= \Big\|\,\text{absgrad}_i \odot \big(\tfrac W2, \tfrac H2\big)\Big\|_2, \qquad
\text{count}_i \mathrel{+}= 1
$$

그리고 이 둘의 비율 `grad2d / count` = **"보였을 때의 평균 화면공간 그래디언트 크기"**가 densification(복제/분할)의 핵심 신호다. 값이 크면 "이 가우시안이 아직 잘 못 맞춘 영역을 덮고 있다"는 뜻이다.

## 코드 흐름 (gsplat `strategy/default.py`)

```python
# 1. 그래디언트 선택: absgrad 또는 grad
if self.absgrad:
    grads = info[self.key_for_gradient].absgrad.clone()   # means2d.absgrad
else:
    grads = info[self.key_for_gradient].grad.clone()

# 2. NDC [-1,1] → 픽셀 단위 스케일
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

# 3. 첫 호출 시 state 초기화 (zeros(N))
#    grad2d, count, (refine_scale2d_stop_iter > 0 이면) radii

# 4. 보인 가우시안만 선택
if packed:
    gs_ids = info["gaussian_ids"]; radii = info["radii"]        # [nnz]
else:
    sel    = info["radii"] > 0.0                                # [C, N]
    gs_ids = torch.where(sel)[1]; grads = grads[sel]; radii = info["radii"][sel]

# 5. 누적
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
if self.refine_scale2d_stop_iter > 0:
    state["radii"][gs_ids] = torch.maximum(state["radii"][gs_ids],
                                           radii / float(max(W, H)))
```

### 각 단계 해설

**(1) `absgrad` vs `grad`** — splatfacto는 `use_absgrad=True`로 `means2d.absgrad`를 쓴다(스텝 D6에서 `retain_grad`, 래스터라이저에 `absgrad=True` 전달). 가우시안 하나의 화면 좌표 $\mu_i^{2D}$는 여러 픽셀에 기여하므로 일반 grad는 픽셀별 기여의 **부호가 상쇄**된다. 양쪽에서 반대 방향으로 당겨지는 가우시안(= 쪼개야 할 것)이 grad≈0으로 보이는 문제가 생긴다. absgrad는 픽셀별 절댓값을 더해 이 신호를 보존한다:

$$
\text{absgrad}_i = \sum_{p} \left|\frac{\partial \mathcal L}{\partial \mu_i^{2D}}\Big|_{p}\right|
$$

**(2) $(W/2, H/2)$ 스케일** — `means2d`의 그래디언트는 NDC $[-1,1]$ 단위다. 픽셀 단위로 바꿔야 `densify_grad_thresh`(splatfacto 기본 0.0008)가 해상도에 무관하게 의미를 갖는다. `n_cameras`도 곱하는데, 배치 렌더 시 loss가 카메라 수로 평균되어 그래디언트가 $1/C$로 줄어드는 것을 보정하는 것. splatfacto는 카메라 1장씩이라 $C=1$.

**(3) 지연 초기화** — `state["grad2d"]`/`count`는 첫 호출 때 `zeros(N)`으로 만들어진다. 이후 refine 단계에서 가우시안이 늘거나 줄면 `_grow_gs`/`_prune_gs`가 state 텐서 크기도 맞춰준다.

**(4) `radii` / `packed` 처리** — 보인 가우시안만 세는 것이 핵심이다.
- `packed=False`(splatfacto 기본): `info["radii"]`는 `[C, N]`(gsplat 버전에 따라 `[C, N, 2]`). `radii > 0`인 (카메라, 가우시안) 쌍만 골라 `gs_ids`로 평탄화한다. 프러스텀 밖이거나 투영 반경이 0인 가우시안은 이번 스텝 통계에 **전혀 반영되지 않는다** — grad2d도, count도.
- `packed=True`: 래스터라이저가 이미 보이는 쌍만 `[nnz]`로 압축해 `gaussian_ids`와 함께 돌려주므로 마스킹 없이 그대로 쓴다.

**(5) 선택적 `radii` 추적** — `refine_scale2d_stop_iter > 0`이면 각 가우시안의 **최대 화면 반경**(`max(W,H)`로 정규화, $[0,1]$)을 `torch.maximum`으로 추적한다. 이는 `_grow_gs`에서 "화면에서 너무 큰 가우시안은 분할", `_prune_gs`에서 "너무 큰 것은 제거"(`prune_scale2d`) 판정에 쓰인다. splatfacto 기본은 0이라 비활성.

## 왜 `grad2d / count`(평균)인가?

- 매 스텝 위치 그래디언트가 크다 = 렌더가 그 자리에서 GT와 잘 맞지 않아 "이동하라"는 압력이 계속 들어온다 = 현재 가우시안 하나로는 그 영역의 디테일을 표현하지 못한다.
- 단순 합(`grad2d`)만 쓰면 자주 보이는 가우시안이 유리해지므로 **보인 횟수 `count`로 나눠** 시점 수에 대한 편향을 없앤다. 한 번만 보였지만 그때 크게 틀린 가우시안과, 100번 보였지만 매번 조금 틀린 가우시안을 공정하게 비교한다.
- `_grow_gs`는 이 평균을 임계값과 비교한다:

$$
\bar g_i = \frac{\text{grad2d}_i}{\text{count}_i} > \tau_g \;(=0.0008)
$$

  통과한 가우시안 중 3D 스케일이 `densify_size_thresh`(0.01)보다 **작으면 복제**(under-reconstruction: 가우시안이 부족), **크면 2개로 분할**(over-reconstruction: 하나가 너무 넓게 덮음).

## 누적 주기와 리셋

`step_post_backward`는 `refine_stop_iter` 전까지 매 스텝 `_update_state`를 호출하고, `refine_every`(100) 스텝마다 `_grow_gs → _prune_gs`를 실행한 뒤 `grad2d`, `count`(그리고 `radii`)를 **0으로 리셋**한다. 즉 판정은 항상 "최근 100스텝 동안의 평균 화면공간 그래디언트"로 이루어진다. 노트북(`splatfacto_train_step.py` I단계)에서 `count > 0`인 가우시안 수와 `grad2d/count` 평균을 임계값과 비교 출력하고, refine 후 `grad2d` sum이 0으로 돌아가는 것을 확인한다.

## 기억 포인트

| 통계 | 누적식 | 의미 |
|---|---|---|
| `grad2d[i]` | `+= ‖absgrad_i · (W/2, H/2)‖₂` (보였을 때만) | 픽셀 단위 화면공간 위치 그래디언트 크기 합 |
| `count[i]` | `+= 1` (보였을 때만) | 이 refine 주기 동안 보인 횟수 |
| `radii[i]` (옵션) | `= max(·, r_i / max(W,H))` | 최대 화면 반경, scale2d 기반 분할/제거용 |
| `grad2d/count` | — | 평균 그래디언트 → 임계값 초과 시 복제/분할 |
