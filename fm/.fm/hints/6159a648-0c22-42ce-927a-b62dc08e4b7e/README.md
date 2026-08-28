# Irradiance Map을 생성하기 위한 렌더링 방정식

## 한 줄 답

$$
L_r(x,\omega_o)=L_e(x,\omega_o)+\int_{\Omega} f_r(x,\omega_i,\omega_o)\, L_i(x,\omega_i)\,(\omega_i \cdot n)\, d\omega_i
$$

이것이 **렌더링 방정식(Rendering Equation, Kajiya 1986)** 이다. 표면의 한 점 $x$에서 방향 $\omega_o$로 나가는 빛(radiance)은, (1) 그 점이 스스로 내는 빛 $L_e$와 (2) 반구 $\Omega$의 모든 방향 $\omega_i$에서 들어온 빛 $L_i$가 BRDF $f_r$에 따라 반사된 양의 합이라는 뜻이다. 이 식을 **360도 전 방향(큐브맵의 모든 픽셀 = 모든 법선 방향 $n$)에 대해 미리 계산해 저장**한 것이 Irradiance Map이다.

## 기호 정리

| 기호 | 의미 |
|---|---|
| $L_r(x,\omega_o)$ | 점 $x$에서 방향 $\omega_o$로 **반사되어 나가는** radiance |
| $L_e(x,\omega_o)$ | 점 $x$가 스스로 **방출**하는 radiance (발광체가 아니면 0) |
| $x$ | 반사가 일어나는 표면 위의 점 |
| $\omega_o$, $\omega_i$ | 빛이 나가는 방향, 빛이 들어오는 방향 (단위 벡터) |
| $\Omega$ | 법선 $n$을 중심으로 한 **반구**(hemisphere) 공간 |
| $f_r(x,\omega_i,\omega_o)$ | BRDF — $\omega_i$로 들어온 빛이 $\omega_o$로 얼마나 반사되는지의 비율 |
| $L_i(x,\omega_i)$ | 방향 $\omega_i$에서 **입사**하는 radiance |
| $(\omega_i \cdot n)$ | 코사인 항 $\cos\theta_i$ — 비스듬히 들어온 빛은 단위 면적당 에너지가 줄어든다 (램버트 코사인 법칙) |
| $d\omega_i$ | 입체각(solid angle) 미소 요소 |

## 왜 이 식이 Irradiance Map이 되는가

Irradiance Map은 이미지를 광원으로 쓰는 **IBL(Image Based Lighting)** 중에서 **Diffuse 반사**를 위해 미리 필터링해 둔 큐브맵이다. 원문에서 소개하는 유도 과정은 다음 세 단계다.

1. **Diffuse 반사 → Lambert BRDF.** Lambert BRDF는 $f_r = \sigma/\pi$ ($\sigma$: 알베도)로 방향과 무관한 **상수**이므로 적분 밖으로 빼낼 수 있다.
2. **표면이 발광하지 않는다** 고 가정하면 $L_e = 0$.

$$
L_r(x,\omega_o)= \frac{\sigma}{\pi} \int_{\Omega} L_i(x,\omega_i)\,(\omega_i \cdot n)\, d\omega_i
$$

3. 여기서 Lambert BRDF를 제외한 적분 부분이 바로 **Irradiance** $E$다.

$$
E(x)= \int_{\Omega} L_i(x,\omega_i)\,(\omega_i \cdot n)\, d\omega_i
$$

핵심 관찰: 단순화된 식에서 $\omega_o$가 사라졌다. 즉 Diffuse 조명은 **보는 방향과 무관**하고 **법선 $n$에만 의존**한다. 그래서 "법선 방향 → 조명값"이라는 함수 하나를 큐브맵에 담아두면, 렌더링 시 `IrradianceMap.Sample(normal)` 한 번으로 Diffuse 조명을 얻을 수 있다. 이것이 Irradiance Map의 존재 이유다.

![환경 큐브맵(좌)과 그것을 컨볼루션해 얻은 Irradiance Map(우)](fig-1.png)

위 그림(learnopengl 출처)에서 왼쪽은 원본 환경 큐브맵(가로수길 HDR), 오른쪽은 같은 환경에 위 적분을 적용한 Irradiance Map이다. 오른쪽 각 픽셀은 "그 방향을 법선으로 갖는 표면이 반구 전체에서 받는 빛의 합"이므로, 나뭇잎·길 같은 고주파 디테일이 전부 사라지고 위쪽(하늘)은 밝고 아래쪽(땅)은 어두운 부드러운 그라데이션만 남는다. 적분(가중 평균)이 저역 통과 필터(low-pass filter) 역할을 한다는 점이 시각적으로 드러난다.

## 실제 계산: 구면 좌표로 바꾸고 리만합으로 근사

셰이더에서 적분을 직접 할 수는 없으므로 원문은 다음과 같이 이산화한다. 알베도 $\sigma$는 렌더링 시 곱하면 되므로 빼고, $1/\pi$만 남긴다.

$$
\frac{1}{\pi}\int_{\Omega} L_i(\omega_i)(\omega_i\cdot n)\,d\omega_i
=\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L(\theta,\phi)\cos\theta\,\sin\theta\,d\theta\,d\phi
$$

- $\cos\theta$ : 식의 $(\omega_i\cdot n)$ 항 (법선과 입사 방향의 각 $\theta$).
- $\sin\theta$ : 입체각 미소 요소 $d\omega = \sin\theta\,d\theta\,d\phi$ 에서 온 야코비안. 구면 위에서 극(pole) 근처는 실제 면적이 작기 때문에 그만큼 가중치를 줄이는 항이다.

리만합으로 바꾸면

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2}\sum_{\phi}\sum_{\theta} L_i\cos\theta\sin\theta
=\frac{\pi}{n_1 n_2}\sum_{\phi}\sum_{\theta} L_i\cos\theta\sin\theta
$$

원문 픽셀 셰이더가 정확히 이것을 수행한다: 법선(`normal = normalize(localPosition)`)을 중심으로 접선 공간을 만들고, `phi`는 $0\sim2\pi$, `theta`는 $0\sim\pi/2$를 `SampleDelta = 0.025` 간격으로 돌며 `CubeMap.Sample(...) * cos(theta) * sin(theta)`를 누적한 뒤 마지막에 `PI * irradiance / numSample`로 정규화한다. 이 결과를 32x32 크기의 작은 큐브맵 여섯 면에 지오메트리 셰이더(`SV_RenderTargetArrayIndex`)로 한 번에 써 넣는다.

| 원본 스카이박스 | 계산된 Irradiance Map |
|---|---|
| ![원본 스카이박스 큐브맵](fig-2.png) | ![Irradiance Map 결과](fig-3.png) |

원문의 실제 결과다. 왼쪽 원본은 구름, 수평선의 밝은 띠, 바다의 질감이 선명하지만, 오른쪽 Irradiance Map에서는 그 모든 것이 뭉개져 위쪽(하늘 방향 법선)은 어두운 남색, 수평선 부근은 밝은 청회색, 아래(바다 방향 법선)는 진한 파랑으로만 남는다. 각 픽셀이 반구 전체의 가중 평균이라 이렇게 될 수밖에 없고, 그래서 32x32 저해상도로도 충분하다. 사용은 `IrradianceMap.Sample(LinearSampler, normal).rgb` 한 줄이다.

## 이 카드가 이어지는 곳

원문의 후반부는 "이 적분 결과가 매우 저주파"라는 성질을 이용해, 24KB짜리 32x32 큐브맵 대신 **구면 조화 함수(SH) 9개 계수 x RGB = 27개 float(108Byte)** 로 같은 Irradiance를 근사하는 방법(Ramamoorthi & Hanrahan)을 다룬다. 즉 이 렌더링 방정식은 SH 기반 IBL의 출발점이다.

## 참고

- Kajiya, "The Rendering Equation", SIGGRAPH 1986
- learnopengl.com, PBR/IBL/Diffuse irradiance
- Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance Environment Maps", SIGGRAPH 2001
