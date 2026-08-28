# 픽셀 셰이더 식 정리에서 알베도 $\sigma$를 제외하는 이유

## 질문
픽셀 셰이더에서 계산할 식을 정리할 때 알베도 $\sigma$를 제외하는 이유는?

## 답
$\sigma$는 **표면 고유 값**(재질 속성)이라 실제 렌더링 시 곱해주면 되기 때문이다. 반면 Lambert BRDF의 $\frac{1}{\pi}$는 **상수**이므로 Irradiance Map 생성 단계에서 미리 함께 계산해 둔다.

---

## 1. 출발점: Diffuse 반사 radiance 식

렌더링 방정식에서 BRDF를 Lambert($f_r = \sigma/\pi$)로 두고, 표면 자체 발광이 없다고 가정하면 다음이 된다.

$$ L_r(\mathbf{x}, \omega_o) = \frac{\sigma}{\pi} \int_{\Omega} L_i(\mathbf{x}, \omega_i)(\omega_i \cdot \mathbf{n})\, d\omega_i $$

- $\sigma$ : 표면의 알베도(diffuse 반사율, 색)
- $\frac{1}{\pi}$ : Lambert BRDF의 에너지 보존 정규화 상수
- 적분 부분 $E(\mathbf{x}) = \int_\Omega L_i (\omega_i \cdot \mathbf{n}) d\omega_i$ : **Irradiance**

식은 세 부분의 곱이다: `[표면 고유값 σ] × [상수 1/π] × [조명 환경에만 의존하는 적분]`.

## 2. 무엇을 미리 계산(precompute)할 수 있는가

Irradiance Map은 **큐브맵 방향 $\mathbf{n}$마다** 적분값을 미리 계산해 텍스처에 저장하는 것이다. 이때 텍스처에 넣을 수 있는 것은 **어느 물체에 적용해도 같은 값**이어야 한다.

| 항 | 의존 대상 | 미리 계산 가능? | 처리 |
|---|---|---|---|
| $\int_\Omega L_i (\omega_i\cdot\mathbf{n}) d\omega_i$ | 환경맵 + 법선 방향 | 가능 (방향별 텍셀) | Irradiance Map에 저장 |
| $\frac{1}{\pi}$ | 없음 (상수) | 가능 | Irradiance Map 생성 시 같이 곱해둠 |
| $\sigma$ | **각 물체/픽셀의 재질** | 불가능 | 렌더링 시 곱함 |

- $\sigma$를 맵에 굽는 순간 그 맵은 **특정 재질 색 전용**이 된다. 빨간 공, 흰 벽, 텍스처가 입혀진 캐릭터가 모두 같은 하늘 아래 있어도 알베도가 다르므로, 각 물체마다 별도 맵을 만들어야 하는 모순이 생긴다. 알베도는 보통 픽셀마다 텍스처(diffuse/albedo map)에서 읽어오기 때문에 더욱 그렇다.
- 반대로 $\frac{1}{\pi}$는 누가 봐도 항상 같은 수이므로, 렌더링 시 매 픽셀마다 나누는 것보다 맵 생성 시 한 번 곱해두는 쪽이 낫다(연산 절약 + 셰이더 단순화).

그래서 원문에서는 픽셀 셰이더에서 계산할 식을 다음처럼 정리한다.

$$ \frac{1}{\pi} \int_{\Omega} L_i(\mathbf{x}, \omega_i)(\omega_i \cdot \mathbf{n})\, d\omega_i
= \frac{1}{\pi} \int_{0}^{2\pi}\!\!\int_{0}^{\pi/2} L(\theta,\phi)\cos\theta \sin\theta\, d\theta\, d\phi $$

리만합으로 바꾸면 $\frac{\pi}{n_1 n_2}\sum\sum L_i \cos\theta \sin\theta$가 되고, 실제 셰이더에서도 `irradiance = PI * irradiance / numSample;`로 $\frac{1}{\pi}$가 이미 반영되어 있다. 어디에도 알베도는 등장하지 않는다.

## 3. 렌더링 시 $\sigma$를 곱하는 위치

원문의 사용 코드가 이를 그대로 보여준다.

```hlsl
float3 ImageBasedLight( float3 normal )
{
    return IrradianceMap.Sample( LinearSampler, normal ).rgb;   // (1/π) * E(n)
}

float4 lightColor = float4( ImageBasedLight( normal ), 1.f ) * MoveLinearSpace( Diffuse );
//                  ^ 미리 계산된 (1/π)·Irradiance                 ^ 여기서 σ(알베도)를 곱함
```

`Diffuse`가 바로 $\sigma$이고, 맵에서 읽은 값에 곱해서 최종 $L_r = \sigma \cdot \frac{1}{\pi} E$가 완성된다.

## 4. 그림으로 보기

| 원본 스카이박스 (환경 $L_i$) | 생성된 Irradiance Map ($\frac{1}{\pi}E(\mathbf{n})$) |
|---|---|
| ![원본 스카이박스 큐브맵](fig-1.png) | ![Irradiance Map 큐브맵](fig-2.png) |

- 왼쪽은 입력인 환경 큐브맵으로, 하늘·구름·바다·수평선의 밝은 띠가 그대로 보인다.
- 오른쪽은 각 방향 $\mathbf{n}$에 대해 반구 적분을 수행한 결과로, 디테일이 사라지고 부드러운 파란 계조만 남는다(위쪽 면은 하늘을 향한 법선의 값이라 어둡고 채도가 높은 파랑, 아래 면은 바다 쪽으로 약간 다른 톤).
- 이 맵에는 **물체 색이 전혀 들어 있지 않다**는 점이 핵심이다. 순수하게 "이 방향의 법선을 가진 표면이 받는 빛의 양(×1/π)"만 담겨 있으므로, 어떤 알베도의 물체에도 재사용할 수 있다. 알베도를 포함시켰다면 이 그림은 물체마다 다시 만들어야 했을 것이다.

## 5. 한 줄 정리

- **미리 계산할 것** = 조명 환경에만 의존하는 것(적분) + 상수($1/\pi$)
- **렌더링 시 곱할 것** = 물체마다 다른 것($\sigma$, 알베도)

이 분리는 나중에 구면 조화 함수로 Irradiance를 9개 계수(108 Byte)로 압축할 때도 그대로 유지된다. SH 계수 역시 조명 환경만 표현하고, 알베도는 여전히 셰이더에서 곱한다.
