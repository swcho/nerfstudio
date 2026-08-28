# `irradiance = PI * irradiance / numSample` 은 왜 필요할까?

한 줄 요약: **적분을 "직사각형 넓이의 합(리만합)"으로 바꾸면 각 직사각형의 밑변 넓이를 곱해 줘야 하는데, 그 밑변 넓이들과 앞에 붙은 $\frac{1}{\pi}$를 한꺼번에 정리한 계수가 $\frac{\pi}{n_1 n_2}$**이고, 코드의 `numSample`이 바로 $n_1 n_2$이기 때문입니다.

---

## 1. 고교 미적분에서 출발: 적분 = 직사각형 넓이의 합

고등학교에서 정적분을 처음 배울 때 이렇게 정의했습니다.

$$
\int_a^b f(x)\,dx \;\approx\; \sum_{k=1}^{n} f(x_k)\,\Delta x, \qquad \Delta x = \frac{b-a}{n}
$$

구간 $[a,b]$를 $n$개로 똑같이 쪼개고, 각 조각에서 함숫값 $f(x_k)$에 **밑변 길이 $\Delta x$**를 곱한 직사각형 넓이를 모두 더하는 것입니다. 이것이 **리만합**입니다. 핵심은 "함숫값만 더하면 안 되고, 꼭 $\Delta x$를 곱해야 한다"는 점입니다. $\Delta x$는 모든 항에 공통이므로 밖으로 빼서

$$
\int_a^b f(x)\,dx \;\approx\; \frac{b-a}{n}\sum_{k=1}^{n} f(x_k)
$$

로 쓸 수 있습니다. **"합을 구한 뒤 $\frac{b-a}{n}$을 곱한다"** — 이 형태를 기억해 두세요.

## 2. 변수가 두 개면? 이중 적분과 이중 리만합

이 셰이더는 반구 위의 모든 방향에서 들어오는 빛을 더합니다. 방향은 두 각도로 표현됩니다.

- $\theta$ (천정각): 표면 법선에서 얼마나 기울어졌는지. $0 \le \theta \le \frac{\pi}{2}$ (반구이므로 90도까지)
- $\phi$ (방위각): 법선 주위로 한 바퀴. $0 \le \phi \le 2\pi$

원문이 계산하려는 식은 다음과 같습니다(알베도 $\sigma$는 렌더링 때 곱하므로 뺐고, Lambert BRDF의 $\frac{1}{\pi}$만 남았습니다).

$$
\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L(\theta,\phi)\cos\theta\,\sin\theta\;d\theta\,d\phi
$$

변수가 두 개라도 아이디어는 같습니다. 1차원에서 "밑변 길이 $\Delta x$"를 곱했듯, 2차원에서는 **작은 직사각형의 넓이 $\Delta\theta\,\Delta\phi$**를 곱합니다.

- $\phi$ 구간 $[0, 2\pi]$를 $n_1$개로 쪼개면 $\Delta\phi = \dfrac{2\pi}{n_1}$
- $\theta$ 구간 $[0, \frac{\pi}{2}]$를 $n_2$개로 쪼개면 $\Delta\theta = \dfrac{\pi/2}{n_2}$

따라서

$$
\frac{1}{\pi}\int\!\!\int L\cos\theta\sin\theta\,d\theta\,d\phi
\;\approx\;
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\cos\theta\sin\theta
$$

## 3. 계수를 정리하면 $\frac{\pi}{n_1 n_2}$

앞에 붙은 상수만 따로 곱해 봅시다.

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi}{2 n_2}
= \frac{1 \cdot 2\pi \cdot \pi}{\pi \cdot n_1 \cdot 2 n_2}
= \frac{\pi}{n_1 n_2}
$$

$\pi$ 하나는 $\frac{1}{\pi}$와 약분되고, 2는 $\frac{1}{2}$와 약분되어 딱 $\frac{\pi}{n_1 n_2}$만 남습니다. 그래서 최종적으로

$$
\text{(우리가 원하는 값)} \;\approx\; \frac{\pi}{n_1 n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\cos\theta\sin\theta
$$

## 4. 코드와 하나씩 대응시키기

```hlsl
float3 irradiance = 0.f;
float numSample = 0.f;
for ( float phi = 0.f; phi < 2.f * PI; phi += SampleDelta )      // n1개 샘플
{
    for ( float theta = 0.f; theta < 0.5f * PI; theta += SampleDelta )  // n2개 샘플
    {
        ...
        irradiance += CubeMap.Sample(...).rgb * cos( theta ) * sin( theta );  // 시그마 안쪽 항
        ++numSample;                                                          // 총 n1*n2번 증가
    }
}
irradiance = PI * irradiance / numSample;   // (π / (n1 n2)) × 합
```

| 수식 | 코드 |
|---|---|
| $\sum_\phi\sum_\theta L\cos\theta\sin\theta$ | 루프 안에서 누적되는 `irradiance` |
| $n_1$ (φ 방향 샘플 수) | 바깥 루프 반복 횟수 |
| $n_2$ (θ 방향 샘플 수) | 안쪽 루프 반복 횟수 |
| $n_1 n_2$ | `numSample` (안쪽 루프가 총 $n_1 \times n_2$번 돌며 1씩 증가) |
| $\dfrac{\pi}{n_1 n_2}\times(\text{합})$ | `PI * irradiance / numSample` |

즉 마지막 줄은 "리만합에서 빠뜨리면 안 되는 밑변 넓이 $\Delta\theta\Delta\phi$"와 "Lambert BRDF의 $\frac{1}{\pi}$"를 한 번에 곱해 주는 줄입니다.

## 5. 왜 `PI`를 *곱하는가*? 헷갈리기 쉬운 부분

Lambert BRDF 때문에 $\frac{1}{\pi}$를 **나눠야** 할 것 같은데 코드는 $\pi$를 **곱합니다**. 이유는 3절에서 봤듯 적분 구간의 크기에서 $\pi^2$가 나오기 때문입니다.

- 반구 방향 전체의 "넓이"(구간 크기) = $2\pi \times \frac{\pi}{2} = \pi^2$
- 여기에 BRDF의 $\frac{1}{\pi}$를 곱하면 $\pi$

그러니 "샘플 평균($\frac{1}{n_1 n_2}\sum$)에 구간 크기 $\pi^2$를 곱해 적분값을 얻고, 다시 $\frac{1}{\pi}$를 곱한다"라고 읽으면 $\pi$가 남는 것이 자연스럽습니다.

## 6. 간단한 검산: 모든 방향에서 빛이 1로 균일하면?

$L \equiv 1$이라 두면 답은 손으로 구할 수 있습니다.

$$
\frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,d\phi
= \frac{1}{\pi}\cdot 2\pi \cdot \left[\frac{\sin^2\theta}{2}\right]_0^{\pi/2}
= \frac{1}{\pi}\cdot 2\pi\cdot\frac{1}{2} = 1
$$

즉 "균일하게 밝기 1인 하늘 아래서 흰 표면은 밝기 1로 보인다"는 직관과 맞습니다. 코드도 `PI * (Σ cosθ sinθ) / numSample`을 계산하면 $n_1, n_2$가 충분히 클 때 1에 가까워집니다. 만약 마지막 줄을 빼먹고 합만 반환하면 결과가 수천 배 커지고, `/ numSample`만 하면(평균만 내면) $\frac{1}{\pi}\cdot\frac{1}{2}\cdot\ldots$ 꼴로 정확히 $\frac{1}{\pi}$배 작아집니다. 이 줄이 정확히 그 차이를 메워 줍니다.

## 7. 덤: 왜 $\sin\theta$가 곱해져 있나?

고교 기하에서 구의 표면을 위도·경도로 나누면 극 근처의 격자 칸은 적도 근처보다 좁습니다. 각도 격자 $d\theta\,d\phi$를 실제 구면 넓이 $d\omega$로 바꿀 때 $d\omega = \sin\theta\,d\theta\,d\phi$가 되어 이 왜곡을 보정합니다. 그래서 피적분함수에 $\cos\theta$(Lambert의 $\omega_i\cdot n$)와 별도로 $\sin\theta$가 함께 붙어 있습니다. 이 항은 루프 안에서 매 샘플에 곱해지므로 마지막 계수 $\frac{\pi}{n_1 n_2}$와는 별개입니다.

---

**정리**: 적분 → 리만합으로 바꾸면 $\Delta\theta\,\Delta\phi = \frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2}$를 곱해야 하고, 여기에 Lambert의 $\frac{1}{\pi}$를 합치면 $\frac{\pi}{n_1 n_2}$. 코드에서 $n_1 n_2$는 `numSample`이므로 `PI * irradiance / numSample`이 됩니다.
