# 구면 조화 함수 Irradiance Map: 적분식을 두 부분으로 나누기

## 질문
구면 조화 함수를 이용한 Irradiance Map 구현에서 적분식을 어떤 두 부분으로 나누는가?

## 핵심 답
Diffuse Irradiance 적분

$$ E = \frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L_i(x,\omega_i)\cos\theta\,\sin\theta\,d\theta\,d\phi $$

를 다음 두 인자로 분리한다.

1. **Radiance 항** — $\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi} L_i(x,\omega_i)\sin\theta\,d\theta\,d\phi$
   적분 범위 $\theta$의 상한을 $\pi/2$(반구)에서 $\pi$(구 전체)로 바꾼 것. 환경맵(큐브맵)의 빛 자체를 나타내는 부분이다.
2. **Clamped cosine 항** — $\max(\cos\theta, 0)$
   $\cos\theta$를 양수 범위로만 제한한 함수. 구 전체 적분과 곱해지면 $\theta > \pi/2$인 반대편 반구에서 값이 0이 되므로, 결과적으로 원래의 반구 적분과 동일해진다.

## 왜 이렇게 나누는가

원래 식은 표면 법선 $\mathbf n$을 기준으로 한 **반구** $\Omega$에 대한 적분이다. 문제는 법선이 픽셀마다 다르기 때문에 반구도 매번 달라진다는 점이다. 반면 구면 조화 함수(SH)는 **구 전체**에서 정의된 기저 함수이므로, SH로 투영하려면 적분 영역이 항상 구 전체여야 한다.

그래서 트릭은 "반구 제한"을 적분 범위에서 빼내어 **함수의 형태**로 옮기는 것이다.
- 적분 범위는 구 전체로 넓히고 (항 1),
- 대신 피적분 함수에 $\max(\cos\theta,0)$을 곱해 반대편 반구의 기여를 0으로 죽인다 (항 2).

이렇게 하면 Irradiance는 두 구면 함수 — 환경 radiance $L(\theta,\phi)$와 clamped cosine $\max(\cos\theta,0)$ — 의 **구면 위 컨볼루션**이 되고, SH 영역에서 컨볼루션은 계수끼리의 곱으로 바뀐다 (푸리에 변환의 컨볼루션 정리와 같은 성질).

## 각 항의 SH 계수

두 항을 각각 SH 기저 $y_l^m$에 투영한다.

- 항 1 (radiance):
  $$ L_{lm} = \int_{0}^{2\pi}\int_{0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\sin\theta\,d\theta\,d\phi $$
  큐브맵을 샘플링하면서 CPU/GPU에서 계산한다. 환경마다 달라지므로 런타임(또는 로딩 시)에 구한다.

- 항 2 (clamped cosine):
  $$ A_l = \int_{0}^{\pi} \max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta $$
  $\cos\theta$는 천정각(zenith angle) $\theta$에만 의존하고 방위각 $\phi$와 무관하므로 $m=0$인 zonal harmonic 계수만 남는다. 환경과 무관한 **상수**이므로 미리 계산해 둘 수 있다.

## 두 항을 다시 합치기

Irradiance의 SH 계수 $E_{lm}$은

$$ E_{lm} = \sqrt{\frac{4\pi}{2l+1}}\,A_l\,L_{lm} = \hat A_l\,L_{lm} $$

$\sqrt{4\pi/(2l+1)}$은 로컬 좌표(법선을 z축으로 둔 좌표)에서 구한 clamped cosine의 계수를 월드 좌표의 임의 법선 방향으로 회전시키기 위한 가중치다. 이렇게 얻은 $E_{lm}$으로 $E(\theta,\phi)=\sum_{l,m}E_{lm}y_l^m(\theta,\phi)$를 복원하면 어떤 법선 방향의 irradiance도 즉시 평가할 수 있다.

미리 계산된 상수 $\hat A_l$:

| $l$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $\hat A_l$ | 3.1415 | 2.0943 | 0.7853 | 0 | -0.1309 | 0 | 0.0490 |

![clamped cosine의 SH 계수 A_l이 l에 따라 급격히 감소하는 그래프](fig-1.png)

위 그림(Ramamoorthi & Hanrahan)은 Lambertian BRDF(= clamped cosine)의 SH 계수를 $l=0\sim20$까지 그린 것이다. $l=0,1,2$에서는 값이 크지만(약 0.88, 1.02, 0.50), $l=3$에서 정확히 0이 되고 $l\ge4$부터는 $\pm0.1$ 이내에서 진동하며 빠르게 0으로 수렴한다. 즉 $\max(\cos\theta,0)$은 매우 저주파 함수이고, 이 항이 radiance 계수 $L_{lm}$에 곱해지기 때문에 환경맵의 고주파 성분은 자동으로 걸러진다. 그래서 $l\le2$의 9개 계수(RGB 3채널 → 27개 float, 108 Byte)만으로 32x32x6 큐브맵(약 24 KB)과 거의 같은 Irradiance Map을 표현할 수 있다.

## 정리

| | 항 1 | 항 2 |
|---|---|---|
| 식 | $\int_{0}^{2\pi}\!\int_{0}^{\pi} L_i \sin\theta\,d\theta\,d\phi$ | $\max(\cos\theta,0)$ |
| 역할 | 환경의 빛 (반구 → 구 전체로 확장) | 반구 제한을 함수로 표현 (반대편 0) |
| SH 계수 | $L_{lm}$ (환경마다 계산) | $A_l$ ($m=0$, 상수, 미리 계산) |
| 결합 | $E_{lm}=\hat A_l L_{lm}$, $\hat A_l=\sqrt{4\pi/(2l+1)}A_l$ | |

## 참고
- Ramamoorthi & Hanrahan, *An Efficient Representation for Irradiance Environment Maps* (SIGGRAPH 2001)
- Ramamoorthi & Hanrahan, *On the relationship between radiance and irradiance* (JOSA A 2001), 4.A절
