# Splatfacto의 `use_bilateral_grid` — 이미지별 ISP 차이 보정

## 질문
Splatfacto의 `use_bilateral_grid` 옵션은 어떤 문제를 해결하는가?

## 핵심 답
'Bilateral Guided Radiance Field Processing'(**BilaRF**, https://bilarfpro.github.io/) 논문의 기법으로,
**학습 이미지마다 다르게 적용된 카메라 ISP 처리(노출, 화이트밸런스, 톤매핑 등)의 차이를
이미지 공간에서 보정**한다. 학습 이미지 수(`num_train_data`)만큼의 학습 가능한 bilateral grid
(`grid_shape` 기본 `(16, 16, 8)`)를 만들어, 카메라 인덱스(`cam_idx`)별로 렌더링 결과에 적용한다.

## 어떤 문제인가?

실제 촬영 데이터셋(특히 스마트폰 사진)은 프레임마다 카메라가 자동으로:

- **자동 노출(AE)** — 밝기가 사진마다 다름
- **자동 화이트밸런스(AWB)** — 색온도가 사진마다 다름
- **톤매핑/감마 등 ISP 후처리** — 비선형 색 변환이 사진마다 다름

을 다르게 적용한다. 3DGS/NeRF는 "같은 3D 지점은 어느 시점에서 봐도 같은 색"이라는
**photometric consistency**를 가정하므로, 이런 이미지별 색 차이는 모델 입장에서 모순된
supervision이 된다. 그 결과 색이 뿌옇게 평균되거나, 모순을 흡수하려고 **플로터(floater)** 같은
가짜 지오메트리가 생긴다.

## BilaRF의 해결책

3D 장면(가우시안들)은 **일관된 하나의 색**을 학습하게 두고, 이미지별 ISP 차이는
**이미지마다 하나씩 붙는 저해상도 bilateral grid**가 흡수하게 분리한다.

- Bilateral grid는 (공간 X, 공간 Y, 밝기/가이던스 W) 3축의 격자에 **아핀 색 변환**을 저장하는
  구조로, 픽셀 위치 + 픽셀 밝기에 따라 부드럽게 보간(slice)된 변환을 렌더 결과에 적용한다.
  기본 `grid_shape=(16, 16, 8)`은 X=16, Y=16, W(가이던스)=8을 의미한다.
- 즉 "렌더링된 진짜 색 → 그 사진의 ISP를 흉내낸 색"으로 바꿔 GT와 비교하므로,
  손실이 ISP 차이 때문에 3D 파라미터를 망가뜨리지 않는다.

## 코드에서의 동작 (asset: `splatfacto.py`)

### 1) 설정 (`SplatfactoModelConfig`)
```python
use_bilateral_grid: bool = False
"""If True, use bilateral grid to handle the ISP changes in the image space. ..."""
grid_shape: Tuple[int, int, int] = (16, 16, 8)
"""Shape of the bilateral grid (X, Y, W)"""
```

### 2) 생성 — 학습 이미지 수만큼 (`populate_modules`)
```python
if self.config.use_bilateral_grid:
    self.bil_grids = BilateralGrid(
        num=self.num_train_data,          # 이미지당 grid 1개
        grid_X=self.config.grid_shape[0], # 16
        grid_Y=self.config.grid_shape[1], # 16
        grid_W=self.config.grid_shape[2], # 8
    )
```
구현은 `nerfstudio.model_components.lib_bilagrid`의 `BilateralGrid`, `slice`,
`color_correct`, `total_variation_loss`를 사용한다.

### 3) 적용 — 학습 시, 카메라 인덱스로 해당 grid 선택 (`get_outputs`)
```python
if self.config.use_bilateral_grid and self.training:
    if camera.metadata is not None and "cam_idx" in camera.metadata:
        rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)
```
`_apply_bilateral_grid`는 `[0,1]` 정규화된 픽셀 xy 격자를 만들어
`slice(bil_grids, rgb, xy, grid_idx=cam_idx)`로 그 이미지 전용 grid를 렌더 RGB에 적용한다.
**training일 때만** 적용된다 — 평가/뷰어 렌더링은 ISP가 제거된 "본연의 색"을 보여준다.

### 4) 정규화 — TV loss (`get_loss_dict`)
```python
if self.config.use_bilateral_grid:
    loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)
```
grid가 공간적으로 급변하지 않도록(진짜 장면 디테일까지 흡수하지 않도록)
total variation 정규화를 건다.

### 5) 별도 optimizer 파라미터 그룹 (`get_param_groups`)
```python
gps["bilateral_grid"] = list(self.bil_grids.parameters())
```

## 함께 알아둘 것: `color_corrected_metrics`

bilateral grid는 학습 시에만 적용되므로, 평가 시 렌더 결과는 GT의 ISP와 어긋나
PSNR이 낮게 나올 수 있다. `color_corrected_metrics: bool = False`를 켜면 평가 전에
`color_correct(predicted_rgb, gt_rgb)`로 GT에 맞춰 색을 정합시킨 `cc_psnr`/`cc_ssim`/`cc_lpips`를
추가로 기록한다. BilaRF나 Zip-NeRF 계열 논문에서 쓰는 평가 프로토콜과 같은 취지다.

## 요약 표

| 항목 | 내용 |
|---|---|
| 해결하는 문제 | 이미지별 ISP(노출·화이트밸런스·톤매핑) 차이로 인한 photometric 불일치 |
| 출처 | BilaRF: 'Bilateral Guided Radiance Field Processing' |
| grid 개수 | 학습 이미지 수(`num_train_data`)만큼, `cam_idx`로 선택 |
| `grid_shape` | 기본 `(16, 16, 8)` = (X, Y, W: 가이던스 축) |
| 적용 시점 | 학습 중 렌더링된 RGB에만 적용 (`self.training`일 때) |
| 정규화 | `10 * total_variation_loss(grids)` |
| 평가 보조 | `color_corrected_metrics`로 `cc_psnr` 등 기록 |
