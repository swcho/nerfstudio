# $L_{lm}$을 리만합으로 바꾸면 어떤 계산식이 되는가?

## 1. 출발점: 구 전체에 대한 이중 적분

원문에서 구면 조화 함수 계수 $L_{lm}$은 다음 적분으로 정의됩니다.

$$
L_{lm}=\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\; d\theta\, d\phi
$$

기호를 하나씩 풀어 봅시다.

| 기호 | 의미 |
|---|---|
| $\theta$ (천정각) | 위쪽(북극)에서부터 잰 각. $0 \le \theta \le \pi$ |
| $\phi$ (방위각) | 수평면에서 한 바퀴 도는 각. $0 \le \phi \le 2\pi$ |
| $L(\theta,\phi)$ | 그 방향에서 들어오는 빛의 세기(Radiance). 큐브맵에서 읽어 옵니다 |
| $y_l^m(\theta,\phi)$ | 구면 조화 함수. "구 위의 기저 함수"라고 생각하면 됩니다 |
| $\sin\theta$ | 구면에서 넓이를 셀 때 붙는 보정 인수(아래 3절) |

즉, "빛의 세기 $\times$ 기저 함수"를 구 표면 전체에 걸쳐 더한 값이 $L_{lm}$입니다. 고교 과정에서 배운 정적분 $\int_a^b f(x)\,dx$가 변수 두 개로 확장된 것뿐입니다.

## 2. 왜 리만합이 필요한가

컴퓨터(GPU 셰이더)는 적분을 기호적으로 풀 수 없습니다. 대신 **구간을 잘게 쪼개 직사각형 넓이를 더하는** 리만합으로 근사합니다. 고교 미적분에서 배운 정의 그대로입니다.

$$
\int_a^b f(x)\,dx \;\approx\; \sum_{k} f(x_k)\,\Delta x, \qquad \Delta x=\frac{b-a}{n}
$$

변수가 둘이면 직사각형이 아니라 작은 "격자 칸" 하나하나의 값을 더합니다.

$$
\int\!\!\int f(\theta,\phi)\,d\theta\,d\phi \;\approx\; \sum_{\phi}\sum_{\theta} f(\theta,\phi)\,\Delta\theta\,\Delta\phi
$$

## 3. $\sin\theta$는 어디서 왔나

$\theta,\phi$ 격자를 똑같은 크기로 쪼개도, 구 표면에서 실제 넓이는 같지 않습니다. 북극 근처에서는 $\phi$가 한 바퀴 돌아도 원이 작고, 적도 근처에서는 원이 큽니다. 위도 $\theta$에서 그 원의 반지름은 $\sin\theta$(반지름 1인 구 기준)이므로, 격자 한 칸의 실제 넓이는

$$
dA = \sin\theta\; d\theta\, d\phi
$$

입니다. 그래서 적분 안에 $\sin\theta$가 붙어 있고, 리만합으로 바꿔도 각 항에 $\sin\theta$가 그대로 남습니다.

## 4. 격자 간격 정하기

- $\phi$ 방향: 전체 길이 $2\pi$를 $n_1$칸으로 → $\Delta\phi=\dfrac{2\pi}{n_1}$
- $\theta$ 방향: 전체 길이 $\pi$를 $n_2$칸으로 → $\Delta\theta=\dfrac{\pi}{n_2}$

이 두 상수는 모든 항에 공통이므로 시그마 밖으로 빼낼 수 있습니다.

## 5. 정리

적분을 리만합으로 바꾸고 상수를 앞으로 모으면

$$
L_{lm}\;\approx\;\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi}{n_2}
\sum_{\phi=0}^{n_1}\sum_{\theta=0}^{n_2} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta
$$

앞의 상수만 계산해 봅시다.

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi}{n_2}
=\frac{2\pi^2}{\pi\, n_1 n_2}
=\frac{2\pi}{n_1 n_2}
$$

따라서 최종 식은

$$
\boxed{\,L_{lm}\;\approx\;\frac{2\pi}{n_1 n_2}\sum_{\phi=0}^{n_1}\sum_{\theta=0}^{n_2} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,}
$$

입니다. 시그마 안은 "샘플 방향의 빛 $\times$ 구면 조화 함수 값 $\times$ $\sin\theta$", 앞의 계수는 "격자 한 칸의 각도 넓이 $\div \pi$"라고 읽으면 됩니다.

## 6. 컴퓨트 셰이더는 이 식을 어떻게 계산하나

원문의 셰이더 코드는 이 식을 그대로 옮긴 것입니다.

- `for phi ... for theta ...` 이중 루프 = 이중 시그마 $\sum_\phi\sum_\theta$
- `sampleDir = (sinθ cosφ, sinθ sinφ, cosθ)` = 각도 $(\theta,\phi)$를 3차원 단위 벡터로 바꾼 것(구면 좌표 → 직교 좌표)
- `radiance = CubeMap.SampleLevel(...)` = $L(\theta,\phi)$
- `ShFunctionL2(sampleDir, y)` = $y_l^m(\theta,\phi)$ 9개($l=0,1,2$)를 한 번에 계산
- `coeffs[i] += radiance * y[i] * sin(theta)` = 시그마 안의 한 항을 누적

루프가 끝나면 누적된 합에 $\dfrac{2\pi}{n_1 n_2}$를 곱합니다. 코드에서는 샘플 개수 `numSample`을 세어 $n_1 n_2$ 역할로 쓰고, 여러 스레드가 나눠 계산한 부분합을 마지막에 합칩니다(리만합은 항의 순서와 무관하게 더해도 결과가 같으므로 병렬화가 가능합니다).

## 7. 한 줄 요약

$d\theta\,d\phi$를 $\dfrac{\pi}{n_2}\cdot\dfrac{2\pi}{n_1}$로 바꾸고 앞의 $\dfrac{1}{\pi}$와 약분하면 $\dfrac{2\pi}{n_1 n_2}\sum\sum L\,y_l^m\sin\theta$ — 컴퓨트 셰이더는 이 이중 합을 루프로 누적한다.
