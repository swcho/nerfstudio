# E단계 `get_loss_dict`의 main_loss 계산 과정

> **Q.** E단계 `get_loss_dict`의 main_loss 계산 과정은?
>
> **A.** GT를 `image/255` 후 학습 해상도로 평균 풀링 축소(`resize_image`)하고 배경 합성(RGB는 no-op)한 뒤,
> $\mathcal L = (1-\lambda)\mathcal L_1 + \lambda(1-\mathrm{SSIM})$, $\lambda = \text{ssim\_lambda} = 0.2$로 계산한다. 즉 **0.8·L1 + 0.2·(1−SSIM)**.

Splatfacto 학습 step(A~F 단계) 중 **E단계**는 렌더 결과 `outputs["rgb"]`와 GT 이미지를 비교해 스칼라 loss를 만드는 지점이다. 소스는 `nerfstudio/models/splatfacto.py`의 `get_loss_dict`(약 652~710행)와 그 보조 함수들이다.

---

## 1. 전체 흐름 한눈에 보기

```
batch["image"] (uint8, [H,W,3])
   │  get_gt_img
   ├─ image.float() / 255.0                 # [0,1] float로 정규화
   ├─ _downscale_if_required → resize_image # 학습 해상도(d배 축소)로 평균 풀링
   │  composite_with_background
   ├─ RGBA(4채널)면 alpha 합성, RGB면 그대로  # 여기선 RGB → no-op
   ▼
gt_img ─┐
        ├─ (mask 있으면 둘 다 mask 곱)
pred_img┘
   ├─ Ll1     = |gt − pred|.mean()
   ├─ simloss = 1 − SSIM(gt, pred)
   ▼
main_loss = (1 − 0.2)·Ll1 + 0.2·simloss  =  0.8·L1 + 0.2·(1−SSIM)
```

---

## 2. 단계별 상세

### 2-1. GT 정규화 — `get_gt_img`

```python
def get_gt_img(self, image):
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    gt_img = self._downscale_if_required(image)
    return gt_img.to(self.device)
```

- 데이터로더가 주는 이미지는 `uint8 [0,255]`. 렌더 출력 `outputs["rgb"]`가 `[0,1]` float이므로 스케일을 맞추기 위해 255로 나눈다.
- 이미 float이면 그대로 통과한다.

### 2-2. 학습 해상도로 축소 — `_downscale_if_required` → `resize_image`

```python
def _get_downscale_factor(self):
    if self.training:
        return 2 ** max(self.config.num_downscales - self.step // self.config.resolution_schedule, 0)
    return 1
```

- Splatfacto는 **coarse-to-fine 해상도 스케줄**을 쓴다. 초반엔 `2^num_downscales`배 축소된 저해상도로 학습하고, `resolution_schedule` step마다 배율이 절반으로 줄어 결국 `d = 1`(원본 해상도)이 된다.
- 렌더(C단계)도 같은 `d`로 축소된 카메라 내부 파라미터로 그리기 때문에, GT도 같은 배율로 줄여야 픽셀 단위 비교가 가능하다.
- 평가(`eval`) 모드에서는 항상 `d = 1`.

```python
def resize_image(image, d):
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
```

- `d×d` 커널의 모든 값이 `1/d²`, `stride=d` → **겹치지 않는 d×d 블록의 평균**. OpenCV의 `INTER_AREA`와 동일한 **평균 풀링(area downsampling)** 이다.
- 채널을 배치 차원으로 옮겨(`[C,1,H,W]`) 채널별로 독립 컨볼루션하고, 다시 `[H/d, W/d, C]`로 되돌린다.
- 보간(bilinear 등) 대신 평균 풀링을 쓰는 이유: 앨리어싱 없이 저역통과가 되고, 렌더 쪽의 저해상도 결과와 통계적으로 잘 맞는다.

### 2-3. 배경 합성 — `composite_with_background`

```python
def composite_with_background(self, image, background):
    if image.shape[2] == 4:
        alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
        return alpha * image[..., :3] + (1 - alpha) * background
    return image
```

- GT가 **RGBA**(알파 채널 포함)일 때만 동작: 렌더러가 빈 공간을 `outputs["background"]` 색으로 채우므로, GT의 투명 영역도 같은 배경색으로 합성해 공정하게 비교한다.
- 일반적인 **RGB 3채널 GT에서는 그대로 반환(no-op)** — 분석 대상 데이터셋이 이 경우다.

### 2-4. (선택) 마스크 적용

```python
if "mask" in batch:
    mask = self._downscale_if_required(batch["mask"]).to(self.device)
    gt_img = gt_img * mask
    pred_img = pred_img * mask
```

- `batch["mask"]`가 있으면 마스크도 같은 배율로 축소한 뒤 GT와 렌더 **둘 다** 곱해 마스크 밖을 검게 만든다. 코드 주석대로 SSIM엔 "약간 sketchy"한 방식(경계에서 인위적 구조가 생김)이다. 이 카드의 상황에서는 mask가 없다.

### 2-5. L1 항

```python
Ll1 = torch.abs(gt_img - pred_img).mean()
```

$$
\mathcal L_1 = \frac{1}{3HW}\sum_{p,c}\big|I_{p,c} - \hat I_{p,c}\big|
$$

- 모든 픽셀·채널의 절대 오차 평균. 픽셀 단위 **색 오차**를 직접 벌점으로 준다. L2(MSE)보다 이상치에 둔감해 초기 학습이 안정적이다.

### 2-6. SSIM 항

```python
simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
```

- `self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)` (`pytorch_msssim`). 기본 11×11 가우시안 윈도우(σ=1.5), $C_1=(0.01)^2$, $C_2=(0.03)^2$.
- 입력을 `[H,W,C]` → `[1,C,H,W]`(NCHW)로 바꿔 넣는다.

$$
\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2 + C_1)(\sigma_x^2+\sigma_y^2 + C_2)}
$$

- 국소 평균·분산·공분산으로 **밝기·대비·구조** 유사도를 측정. SSIM은 1이 완전 일치이므로 `1 − SSIM`을 loss로 쓴다. L1이 놓치는 텍스처·엣지의 흐릿함을 잡아준다.

### 2-7. 가중 합 — main_loss

```python
"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
```

$$
\mathcal L = (1-\lambda)\,\mathcal L_1 + \lambda\,(1 - \mathrm{SSIM}),\qquad \lambda = \texttt{ssim\_lambda} = 0.2
$$

→ **`0.8·L1 + 0.2·(1−SSIM)`**. 이 λ=0.2는 원 논문(3D Gaussian Splatting, Kerbl et al. 2023)의 $\mathcal L = (1-\lambda)\mathcal L_1 + \lambda\mathcal L_{D\text{-}SSIM}$, $\lambda=0.2$를 그대로 따른 값이다.

---

## 3. main_loss 외에 `loss_dict`에 함께 들어가는 항들

카드의 초점은 `main_loss`지만, 실제 `get_loss_dict`는 여러 항을 반환하고 Pipeline/Trainer가 `sum(loss_dict.values())`로 합친다.

| 키 | 조건 | 내용 |
|---|---|---|
| `main_loss` | 항상 | 위의 0.8·L1 + 0.2·(1−SSIM) |
| `scale_reg` | `use_scale_regularization=True` **이고** `step % 10 == 0` | 가우시안의 (최대 축 / 최소 축) 비율이 `max_gauss_ratio`(=10)를 넘는 만큼 벌점, `0.1 × mean`. 바늘처럼 길쭉한 가우시안 억제. 기본값은 False → `0.0` 텐서 |
| `mcmc_opacity_reg`, `mcmc_scale_reg` | `strategy == "mcmc"` | MCMC 전략용 opacity/scale L1 정규화 |
| camera optimizer 항 | `self.training` | `camera_optimizer.get_loss_dict`가 카메라 pose 보정량(translation/rotation)에 대한 정규화 항을 추가 (`camera_opt_regularizer` 등) |
| `tv_loss` | `use_bilateral_grid=True` | bilateral grid의 total variation × 10 |

기본 설정(default strategy, scale_reg 꺼짐, bilateral grid 꺼짐)에서 실질적으로 그래디언트를 만드는 항은 **main_loss + camera optimizer 정규화** 정도다.

---

## 4. `get_metrics_dict`와의 관계

E단계에서 `get_loss_dict` 직전에 호출되는 `get_metrics_dict`도 **같은 방식으로 GT를 준비**(`composite_with_background(get_gt_img(...))`)한 뒤 **PSNR**(MSE 기반, 모니터링용)과 `gaussian_count`를 기록한다. 즉 GT 전처리 파이프라인은 metrics/loss 양쪽에서 동일하다.

$$
\mathrm{PSNR} = -10\log_{10}\Big(\frac{1}{3HW}\sum_{p,c}(I_{p,c}-\hat I_{p,c})^2\Big)
$$

---

## 5. 기억 포인트

1. **GT 전처리 3단계**: `/255` → 평균 풀링 축소(`resize_image`, 학습 해상도 `d`) → 배경 합성(RGB면 no-op).
2. **두 항**: L1(픽셀 색 오차) + (1−SSIM)(국소 구조 오차).
3. **비율**: `ssim_lambda = 0.2` → **0.8·L1 + 0.2·(1−SSIM)**, 3DGS 원 논문과 동일.
4. 부가 항(`scale_reg`, camera optimizer, mcmc, tv)은 별도 키로 들어가고 Trainer가 모두 더한다.
