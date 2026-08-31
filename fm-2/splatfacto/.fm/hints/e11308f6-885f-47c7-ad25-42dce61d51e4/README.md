# `use_scale_regularization` — PhysGaussian의 스케일 정규화

## 카드 요약

**Q.** `use_scale_regularization` 옵션은 어떤 논문의 기법이며 무엇을 막는가?

**A.** PhysGaussian 논문에서 도입된 스케일 정규화로, 길쭉하게 튀는(spikey) 거대 가우시안을 억제한다. 최대/최소 스케일 비율이 `max_gauss_ratio`(기본 10.0)를 넘는 만큼을 페널티로 부과하며, 10스텝마다 0.1 가중치로 적용된다.

---

## 어디서 나온 기법인가 — PhysGaussian

`splatfacto.py`의 config 주석이 출처를 명시한다:

```python
use_scale_regularization: bool = False
"""If enabled, a scale regularization introduced in PhysGauss
(https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
max_gauss_ratio: float = 10.0
"""threshold of ratio of gaussian max to min scale before applying regularization
loss from the PhysGaussian paper
"""
```

**PhysGaussian**(Xie et al., CVPR 2024)은 3D Gaussian Splatting으로 재구성한 장면에 MPM(Material Point Method) 기반 연속체 역학을 적용해 물리 시뮬레이션을 하는 논문이다. 가우시안을 물리적 "입자"처럼 다뤄 변형시키기 때문에, 렌더링에서는 잘 안 보이던 **극단적으로 길쭉한(anisotropic) 가우시안**이 물체가 움직이거나 변형될 때 표면 밖으로 가시처럼 튀어나오는(spikey) 아티팩트가 된다. 이를 막기 위해 학습 단계에서 가우시안의 3축 스케일이 너무 비대칭해지지 않도록 하는 정규화 손실을 제안했고, splatfacto가 이를 옵션으로 가져왔다.

## 무엇을 막는가

3DGS의 각 가우시안은 3개의 축별 스케일 \(s_1, s_2, s_3\)을 갖는다. 재구성 손실만으로 학습하면 사진만 잘 맞추면 되므로, 바늘·판자처럼 한 축만 극단적으로 긴 가우시안이 생기기 쉽다. 이런 가우시안은:

- 특정 시점에서는 얇게 보여 문제가 없지만, 다른 시점·다른 해상도에서 **스파이크/침 모양 아티팩트**로 보인다.
- 물리 시뮬레이션·메시 추출·편집 등 다운스트림 작업에서 특히 문제가 된다.

스케일 정규화는 **최대 스케일 / 최소 스케일 비율**(이방성 비율)이 임계값을 넘지 못하게 눌러서, 가우시안을 상대적으로 "통통한" 형태로 유지한다.

## 실제 구현 (`get_loss_dict`)

```python
if self.config.use_scale_regularization and self.step % 10 == 0:
    scale_exp = torch.exp(self.scales)
    scale_reg = (
        torch.maximum(
            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
            torch.tensor(self.config.max_gauss_ratio),
        )
        - self.config.max_gauss_ratio
    )
    scale_reg = 0.1 * scale_reg.mean()
else:
    scale_reg = torch.tensor(0.0).to(self.device)

loss_dict = {
    "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
    "scale_reg": scale_reg,
}
```

단계별로 뜯어보면:

1. **`torch.exp(self.scales)`** — 스케일 파라미터는 로그 공간에 저장되므로 exp로 실제 스케일로 변환.
2. **`amax / amin`** — 가우시안마다 3축 스케일 중 최대값/최소값의 비율을 계산. 완전한 구라면 1, 길쭉할수록 커진다.
3. **`torch.maximum(ratio, max_gauss_ratio) - max_gauss_ratio`** — 힌지(hinge) 형태의 페널티. 비율이 10.0 이하면 페널티가 정확히 0이고, 10.0을 **넘는 만큼만** 선형으로 벌점을 준다. 즉 적당한 이방성은 허용하고 극단적인 경우만 처벌한다.
4. **`0.1 * mean()`** — 전체 가우시안 평균에 0.1 가중치를 곱해 `scale_reg` 손실로 추가.

수식으로는:

$$L_{aniso} = 0.1 \cdot \frac{1}{|\mathcal{P}|}\sum_{p} \max\!\left(\frac{\max(S_p)}{\min(S_p)},\ r\right) - r, \qquad r = 10$$

(PhysGaussian 논문의 anisotropy regularizer와 같은 형태.)

## 기억할 디테일

| 항목 | 값 | 비고 |
|---|---|---|
| 출처 | PhysGaussian (CVPR 2024) | MPM 물리 시뮬 논문의 anisotropy loss |
| 기본값 | `use_scale_regularization = False` | 옵트인 옵션 |
| 임계 비율 | `max_gauss_ratio = 10.0` | max/min 스케일 비율의 허용 상한 |
| 적용 주기 | `self.step % 10 == 0` | **10스텝마다 한 번**만 계산 (그 외 스텝은 0) |
| 가중치 | `0.1` | 하드코딩된 계수 |
| 형태 | 힌지 페널티 | 임계값 이하는 벌점 0, 초과분만 선형 페널티 |

참고로 같은 파일의 `mcmc_scale_reg`(MCMC 전략 전용, `|exp(scales)|` 평균에 대한 L1 페널티)와는 별개의 정규화다 — MCMC 쪽은 가우시안이 전반적으로 커지는 것을 누르는 것이고, 이 카드의 스케일 정규화는 **비율(이방성)**을 누르는 것이다.

소스: `fm-2/splatfacto/.fm/assets/splatfacto.py` (config 133–138행, 손실 계산 675–691행)
