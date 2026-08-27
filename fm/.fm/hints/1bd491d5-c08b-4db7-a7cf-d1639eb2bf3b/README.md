# D단계 `model.get_outputs(camera)` — D1~D7 하위 단계

> **질문**: D단계 `model.get_outputs(camera)`를 펼친 D1~D7 하위 단계는?
>
> **답**: D1 카메라 보정 → D2 해상도 스케줄 → D3 viewmat/K 계산 → D4 SH 차수 결정 → D5 gsplat rasterization → D6 `step_pre_backward`(means2d.retain_grad) → D7 배경 합성.

Splatfacto 학습 스텝(A~I) 중 **D는 forward 렌더링** 단계입니다. `nerfstudio/models/splatfacto.py`의 `get_outputs`(약 485~600행)는 카메라 하나를 받아 가우시안을 화면에 투영하고 `rgb`, `alpha`, `background` 등을 돌려줍니다. 이 함수를 순서대로 펼치면 아래 7단계가 됩니다.

## 한눈에 보기

| 단계 | 이름 | 핵심 코드 | step 0 기준 값 |
|---|---|---|---|
| D1 | 카메라 보정 | `camera_optimizer.apply_to_camera(camera)` | mode="off" → 보정량 0 |
| D2 | 해상도 스케줄 | `d = 2^max(num_downscales − step // resolution_schedule, 0)` | d = 4 |
| D3 | viewmat / K | `get_viewmat(c2w)` [1,4,4], `get_intrinsics_matrices()` [1,3,3] | K는 1/d 로 축소 |
| D4 | SH 차수 | `min(step // sh_degree_interval, sh_degree)` | 0 (DC 밴드만) |
| D5 | 래스터화 | `gsplat.rasterization(...)` → `render, alpha, info` | 활성함수 exp/sigmoid 적용 |
| D6 | pre-backward | `strategy.step_pre_backward(...)` → `means2d.retain_grad()` | 화면공간 grad 보존 |
| D7 | 배경 합성 | `rgb = clamp(render + (1−alpha)·background)` | background ~ U[0,1]³ |

암기 팁: **"카메라 → 해상도 → 행렬 → SH → 래스터 → 훅 → 배경"**. 앞의 넷(D1~D4)은 래스터화 *입력 준비*, D5가 *본체*, 뒤의 둘(D6~D7)은 *후처리*입니다.

## 단계별 설명

### D1. 카메라 보정
`self.training`일 때만 `camera_optimizer.apply_to_camera(camera)`로 포즈 보정을 적용한 `optimized_camera_to_world`([1,3,4])를 얻습니다. 기본 설정 `mode="off"`이면 원본 `camera.camera_to_worlds`와 완전히 같습니다(노트북에서 보정량 = 0 확인). 학습이 아닐 때는 그대로 사용하며, 이 시점에 `crop_box`가 있으면 가우시안 crop도 처리합니다.

### D2. 해상도 스케줄
`_get_downscale_factor()`가 `d = 2^max(num_downscales − step // resolution_schedule, 0)`을 계산합니다. 초기에는 저해상도(step 0 → d=4, 즉 1/4 크기)로 렌더링해 빠르고 거친 형태부터 맞추고, 스텝이 진행되면 d가 2, 1로 줄어 원본 해상도로 수렴합니다(coarse-to-fine).

### D3. viewmat / K 계산
카메라를 잠시 `rescale_output_resolution(1/d)`로 축소해 `K`와 `W, H`를 뽑고 다시 `rescale_output_resolution(d)`로 원복합니다.
- `viewmat = get_viewmat(c2w)`: 포즈 $c2w=[R|t]$를 뒤집어 world→camera 변환 $\begin{bmatrix}R^\top & -R^\top t\\0&1\end{bmatrix}$을 만듭니다. 이때 nerfstudio(OpenGL, −z 전방)와 gsplat(OpenCV, +z 전방) 규약 차이를 y, z 축 부호 반전으로 맞춥니다.
- `K`: $\frac1d\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&d\end{bmatrix}$ — 초점거리와 주점이 d로 줄어든 내부행렬.

### D4. SH 차수 결정
색은 구면조화(SH) 계수 `[N,16,3]`(`features_dc` 1개 + `features_rest` 15개)로 저장됩니다. 학습 초기에는 `sh_degree_to_use = min(step // sh_degree_interval(1000), sh_degree(3))`로 낮은 차수만 사용 → step 0에서는 0차(뷰 무관 색)만, 1000스텝마다 한 차수씩 열어 최대 3차(16밴드)까지 사용합니다. `sh_degree == 0` 설정이면 SH 대신 `sigmoid(colors)`를 직접 색으로 씁니다.

### D5. gsplat rasterization
`gsplat.rendering.rasterization`을 호출합니다. 파라미터의 활성함수는 **여기서** 적용됩니다: `scales=exp(scales)`, `opacities=sigmoid(opacities)`, quats는 gsplat 내부에서 정규화. 주요 인자: `viewmats=viewmat, Ks=K, width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB"(학습) / "RGB+ED"(평가 또는 depth 출력 시), sh_degree, absgrad=strategy.absgrad, rasterize_mode("classic"/"antialiased")`.
출력은 `render [1,H,W,3]`, `alpha [1,H,W,1]`, 그리고 `info` 딕셔너리(`radii`, `means2d`, `width/height/n_cameras` 등)로, `info`는 `self.info`에 저장되어 이후 I단계(densification)에서 쓰입니다.
수식으로는 각 가우시안의 3D 공분산 $\Sigma_i=RSS^\top R^\top$을 화면에 투영한 2D 가우시안으로 깊이 순 알파 블렌딩: $\text{render}(p)=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$, $\text{alpha}(p)=1-\prod_i(1-\alpha_i)$.

### D6. `step_pre_backward` (means2d.retain_grad)
학습 중이면 `strategy.step_pre_backward(gauss_params, optimizers, strategy_state, step, info)`를 호출합니다. `DefaultStrategy`의 이 함수는 `info["means2d"].retain_grad()`를 걸어, backward 이후에도 화면공간 2D 위치에 대한 그래디언트(`means2d.grad` 또는 `absgrad=True`일 때 `means2d.absgrad`)가 남도록 합니다. 이 값이 I단계에서 split/duplicate(densification) 판단의 기준이 됩니다. 절댓값 누적(absgrad)을 쓰면 서로 반대 방향의 그래디언트가 상쇄되지 않아 판단이 더 안정적입니다.

### D7. 배경 합성
`background = _get_background_color()` — 학습 중 기본값은 `torch.rand(3)` 랜덤 색입니다. `rgb = clamp(render + (1−alpha)·background, 0, 1)`로 투과율이 남은 픽셀(빈 공간)에 배경색을 채웁니다. 매 스텝 배경이 바뀌므로 가우시안이 "배경색을 흉내내어" 빈 공간을 덮어버리는 국소해가 억제됩니다. 이후 옵션으로 bilateral grid 적용, `RGB+ED`이면 depth 추출을 거쳐 `{"rgb", "depth", "accumulation"(alpha), "background"}`를 반환합니다.

## 검증 (노트북)
`splatfacto_train_step.py`의 D절은 D1~D7을 손으로 실행한 `rgb_manual`과, 같은 시드(`torch.manual_seed(123)` → 같은 랜덤 배경)로 호출한 `model.get_outputs(camera)["rgb"]`를 픽셀 단위로 비교해 최대 오차가 ~0 임을 확인합니다. 이후 단계(E~I)는 retain_grad 훅이 걸린 원본 호출의 `model.info`를 사용합니다.

## 참고
- `nerfstudio/models/splatfacto.py` — `get_outputs`, `get_viewmat`, `_get_downscale_factor`, `_get_background_color`
- `.fm/assets/splatfacto_train_step.py` — D절(forward)
- gsplat `DefaultStrategy.step_pre_backward` — `means2d.retain_grad()`
