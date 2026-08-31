# `sh_degree=0`일 때 Splatfacto의 색상 표현 변화

## 질문
`sh_degree=0`으로 설정하면 색상 표현이 어떻게 달라지는가?

## 핵심 답
SH(spherical harmonics) 계수 대신 **가우시안당 RGB 색상 하나만 직접 최적화하는 모드**가 된다. `features_dc`는 SH의 DC 계수가 아니라 "logit 공간의 RGB"로 재해석되며, 렌더링 시 `sigmoid(features_dc)`로 [0,1] 범위의 RGB를 얻는다. 뷰 방향에 따라 색이 변하는 view-dependent 효과(반사광, 하이라이트 등)는 표현할 수 없게 되는 대신 파라미터 수와 계산량이 줄어든다.

## 코드에서의 세 갈래 분기 (`splatfacto.py`)

### 1. 초기화 — `populate_modules()` (약 209–214행)

SfM seed point의 색상(0–255)을 파라미터로 저장할 때 변환 방식이 달라진다.

```python
if self.config.sh_degree > 0:
    shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)   # SH DC 계수로 변환
    shs[:, 1:, 3:] = 0.0
else:
    CONSOLE.log("use color only optimization with sigmoid activation")
    shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)  # logit 공간 RGB
```

- `sh_degree > 0`: `RGB2SH(rgb) = (rgb - 0.5) / C0` (C0 ≈ 0.2820948)로 0차 SH 계수를 만든다. SH basis의 0차 항이 상수이므로 이 계수가 평균 색을 담당한다.
- `sh_degree == 0`: 나중에 sigmoid를 씌워 RGB를 복원할 것이므로, sigmoid의 역함수인 `torch.logit`으로 저장한다(`eps=1e-10`은 0/1 근처에서의 무한대 방지). 즉 `sigmoid(logit(c)) = c`가 되도록 초기화하는 것.

또한 `dim_sh = num_sh_bases(sh_degree)`가 1이 되므로 `features_rest`는 `(N, 0, 3)` 크기의 빈 텐서가 되어 사실상 `features_dc`(N×3)만 색상 파라미터로 남는다.

### 2. 색상 조회 — `colors` 프로퍼티 (약 297–309행)

```python
@property
def colors(self):
    if self.config.sh_degree > 0:
        return SH2RGB(self.features_dc)          # SH DC → RGB (선형 변환)
    else:
        return torch.sigmoid(self.features_dc)   # logit → RGB (sigmoid)
```

`shs_0` 프로퍼티도 반대로 동작한다: `sh_degree == 0`이면 export 등을 위해 `RGB2SH(sigmoid(features_dc))`로 SH 형식으로 되돌려준다.

### 3. 렌더링 — `get_outputs()` (약 549–569행)

```python
if self.config.sh_degree > 0:
    sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
else:
    colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
    sh_degree_to_use = None

render, alpha, self.info = rasterization(
    ...,
    colors=colors_crop,
    sh_degree=sh_degree_to_use,
    ...
)
```

gsplat의 `rasterization()`은 `sh_degree` 인자로 색상 해석 방식을 구분한다.

- `sh_degree`가 정수면 `colors`를 SH 계수 `[N, K, 3]`으로 받아 카메라 방향에 따라 SH를 평가한다. (참고로 `sh_degree > 0`일 때는 `sh_degree_interval`(기본 1000 스텝)마다 사용하는 차수를 하나씩 늘리는 coarse-to-fine 스케줄도 적용된다.)
- **`sh_degree=None`이면 `colors`를 이미 계산된 RGB `[N, 3]`으로 받아 그대로 사용한다.** 그래서 rasterizer에 넘기기 전에 sigmoid를 먼저 적용하고 `[N, 1, 3]`을 `[N, 3]`으로 squeeze한다.

## 왜 sigmoid/logit인가?

SH 계수는 범위 제한 없이 자유롭게 최적화해도 되지만, 색상을 직접 최적화할 때는 RGB가 [0,1] 안에 있어야 한다. 파라미터 자체를 clamp하는 대신 **제약 없는 logit 공간에서 최적화하고 sigmoid로 [0,1]에 매핑**하는 표준 reparameterization을 쓴다(opacity가 `torch.logit(0.1 * ones)`로 초기화되고 `sigmoid(opacities)`로 사용되는 것과 동일한 패턴).

## 정리 표

| 단계 | `sh_degree > 0` | `sh_degree == 0` |
|---|---|---|
| 초기화 | `RGB2SH(rgb)` → SH DC 계수 | `torch.logit(rgb)` → logit 공간 RGB |
| 파라미터 | `features_dc`(DC) + `features_rest`(고차 SH) | 사실상 `features_dc`(N×3)만 |
| 색상 복원 | `SH2RGB(features_dc)` | `torch.sigmoid(features_dc)` |
| rasterization 호출 | `sh_degree=사용 차수` (계수 전달, 뷰 방향으로 SH 평가) | `sh_degree=None` (sigmoid 적용된 RGB 직접 전달) |
| view-dependent 색 | 표현 가능 | 불가 (뷰와 무관한 단일 색) |

## 암기 포인트
- `sh_degree=0` = "color only optimization with sigmoid activation" (코드에 로그로 그대로 찍힘).
- 3종 세트: **초기화는 logit, 사용은 sigmoid, rasterization은 `sh_degree=None`**.
