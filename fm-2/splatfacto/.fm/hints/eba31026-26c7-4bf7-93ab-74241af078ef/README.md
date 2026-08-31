# nerfstudio의 Splatfacto 모델은 무엇인가?

**한 줄 요약**: Splatfacto는 nerfstudio가 구현한 3D Gaussian Splatting(3DGS) 모델이다. 원조 3DGS(Kerbl et al., SIGGRAPH 2023) 이후 나온 여러 후속 기법 — absgrad(AbsGS), MCMC 밀도화 전략(3DGS-MCMC), antialiased 래스터라이즈(Mip-Splatting), bilateral grid(BilaRF) 등 — 을 하나의 모델로 통합했고, 실제 래스터라이제이션 연산은 **gsplat 라이브러리**에 위임한다.

소스: `splatfacto.py` (772줄). 모듈 docstring부터 정체성을 선언한다:

```python
"""
Gaussian Splatting implementation that combines many recent advancements.
"""
```

---

## 1. 배경: 3D Gaussian Splatting이란

3DGS는 장면을 수십만~수백만 개의 **3D 가우시안**(위치 mean, 공분산 = scale×rotation, 불투명도 opacity, 뷰 의존 색상 SH 계수)으로 표현하고, 이를 화면에 투영(splatting)해 알파 블렌딩으로 렌더링하는 기법이다. NeRF처럼 레이마칭으로 MLP를 수백 번 질의하는 대신, 명시적 프리미티브를 GPU 래스터라이저로 그리므로 **학습·렌더링 모두 훨씬 빠르다**. 학습 중에는 가우시안을 복제(duplicate)/분할(split)/제거(cull)하는 **densification(밀도화)** 과정으로 장면의 디테일에 맞게 프리미티브 수를 조절한다.

Splatfacto라는 이름은 nerfstudio의 명명 관례(nerfacto = NeRF + de facto)를 따른 것으로, "Gaussian Splatting의 nerfstudio 기본(de facto) 구현"이라는 뜻이다.

## 2. gsplat에 위임하는 구조

Splatfacto 자체는 CUDA 커널을 갖고 있지 않다. 렌더링과 밀도화 전략 모두 gsplat(nerfstudio 팀이 관리하는 오픈소스 라이브러리)에서 가져온다:

```python
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.rendering import rasterization   # gsplat>=1.0.0 필요
```

`get_outputs()`의 핵심은 결국 `rasterization(...)` 한 번의 호출이다(555행). 모델이 하는 일은 파라미터 관리(`gauss_params` ParameterDict: means/scales/quats/features_dc/features_rest/opacities), 카메라 행렬 변환(`get_viewmat` — c2w를 gsplat 규약의 world2camera로 변환, z/y축 뒤집기), 손실 계산, 학습 콜백 연결이다.

- **활성화 함수 규약**: scale은 `torch.exp`, opacity는 `torch.sigmoid`를 거쳐 rasterization에 전달된다(파라미터는 log/logit 공간에 저장).
- **초기화**: SfM 포인트(seed_points)가 있으면 그 위치·색으로, 없으면 랜덤 초기화(`random_init`). 각 점의 초기 scale은 3-최근접 이웃 평균 거리(`k_nearest_sklearn`)로 정한다.

## 3. 통합된 최신 기법들 (config 옵션으로 선택)

### (a) absgrad — AbsGS (기본 켜짐)
```python
use_absgrad: bool = True
```
원조 3DGS는 화면공간 위치 gradient의 **합**으로 밀도화 대상을 골랐는데, 여러 뷰의 gradient가 서로 상쇄되어 커야 할 gradient가 작게 나오는 문제가 있다. AbsGS(2024)는 gradient의 **절댓값 합(homodirectional gradient)** 을 기준으로 써서 과소분할(over-reconstruction)로 인한 블러를 줄인다. `DefaultStrategy(absgrad=...)`와 `rasterization(absgrad=...)` 양쪽에 전달된다.

### (b) MCMC 전략 — 3DGS-MCMC
```python
strategy: Literal["default", "mcmc"] = "default"
max_gs_num: int = 1_000_000   # cap_max
noise_lr: float = 5e5
```
3DGS-MCMC(NeurIPS 2024)는 휴리스틱한 clone/split/prune 대신, 가우시안 집합을 **SGLD(확률적 경사 랑주뱅 동역학) 샘플**로 해석한다. means 업데이트에 노이즈를 주입하고, 죽은(저불투명도) 가우시안을 살아있는 가우시안 위치로 **재배치(relocation)** 하며, 총 개수를 `cap_max`로 상한 관리한다. 코드에서는 `MCMCStrategy` 사용 시 `step_post_backward`에 means의 현재 학습률을 넘기고(383행), 손실에 opacity/scale L1 정규화(`mcmc_opacity_reg`, `mcmc_scale_reg`)를 추가한다(694~702행).

### (c) antialiased 래스터라이즈 — Mip-Splatting 계열
```python
rasterize_mode: Literal["classic", "antialiased"] = "classic"
```
classic 모드는 EWA splatting에 고정된 화면공간 블러 커널([0.3, 0.3])을 더하는데, 학습 해상도와 다른 해상도로 렌더링하면 앨리어싱류 아티팩트가 생긴다. antialiased 모드는 블러로 인해 변한 적분 밀도를 보정하는 **compensation factor를 opacity에 곱해** 이를 해결한다(Mip-Splatting, CVPR 2024의 2D Mip 필터 아이디어). 단, config 주석이 경고하듯 antialiased로 export한 PLY는 classic 전용 웹 뷰어와 호환되지 않는다.

### (d) bilateral grid — BilaRF
```python
use_bilateral_grid: bool = False
grid_shape: Tuple[int, int, int] = (16, 16, 8)
```
"Bilateral Guided Radiance Field Processing"(SIGGRAPH 2024)에서 온 기법. 사진마다 다른 카메라 ISP 처리(노출·화이트밸런스·톤매핑)를 **이미지별 학습 가능한 bilateral grid**로 흡수해서, 3D 표현이 ISP 불일치를 플로터로 설명하려는 것을 막는다. 학습 시 렌더 결과에 `_apply_bilateral_grid`로 grid를 slice해 적용하고, TV(total variation) 손실 `10 * total_variation_loss(...)`로 grid를 매끄럽게 유지한다. 관련 옵션 `color_corrected_metrics`는 평가 시 색 보정 후 PSNR/SSIM/LPIPS(cc_psnr 등)를 함께 기록한다.

### (e) 그 외 통합 요소
- **카메라 포즈 최적화**: `camera_optimizer` (기본 off) — 학습 중 카메라 외부 파라미터를 미세 조정.
- **PhysGaussian scale 정규화**: `use_scale_regularization` — max/min scale 비율이 `max_gauss_ratio`(10.0)를 넘는 뾰족한(spiky) 가우시안에 페널티.
- **coarse-to-fine 해상도 스케줄**: 처음엔 1/2^`num_downscales` 해상도로 학습, `resolution_schedule`(3000) 스텝마다 2배씩 올림.
- **SH degree 점진 활성화**: `sh_degree_interval`(1000) 스텝마다 사용 SH 차수를 하나씩 늘려 최대 `sh_degree`(3)까지.

## 4. 학습 손실

```python
main_loss = (1 - ssim_lambda) * L1 + ssim_lambda * (1 - SSIM)   # ssim_lambda = 0.2
```
원조 3DGS와 동일한 L1 + D-SSIM 조합(λ=0.2). 여기에 조건부로 scale_reg, mcmc 정규화, 카메라 옵티마이저 손실, bilateral grid TV 손실이 붙는다. 평가 지표는 PSNR/SSIM/LPIPS.

## 5. 학습 루프 연결 방식

nerfstudio의 콜백 시스템으로 gsplat 전략을 끼워 넣는다:

- `BEFORE_TRAIN_ITERATION` → `step_cb`: 현재 step과 옵티마이저/스케줄러 참조 저장
- 렌더링 중(`get_outputs`) → `strategy.step_pre_backward`: gradient 추적 준비
- `AFTER_TRAIN_ITERATION` → `step_post_backward`: rasterization이 돌려준 `self.info`(화면공간 gradient 통계 등)를 이용해 densify/cull(Default) 또는 relocation+노이즈 주입(MCMC) 수행

주요 기본값: warmup 500스텝 후 100스텝마다 refinement, `densify_grad_thresh=0.0008`, opacity<0.1 컬링, 30 refinement 주기마다 alpha 리셋, 15000스텝에서 분할 중단.

## 6. 정리

| 질문 포인트 | 답 |
|---|---|
| 정체 | nerfstudio의 3D Gaussian Splatting 구현 (기본 GS 모델) |
| 특징 | 원조 3DGS + 후속 기법 통합: absgrad(AbsGS), MCMC(3DGS-MCMC), antialiased(Mip-Splatting 계열), bilateral grid(BilaRF), 카메라 최적화, PhysGaussian scale 정규화 |
| 래스터라이제이션 | 직접 구현하지 않고 **gsplat** 라이브러리의 `rasterization()`과 `DefaultStrategy`/`MCMCStrategy`에 위임 |
| 손실 | L1 + D-SSIM (λ=0.2) + 옵션 정규화 항들 |
