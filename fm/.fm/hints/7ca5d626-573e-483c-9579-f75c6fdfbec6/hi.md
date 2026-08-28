# 반구 적분을 구면 좌표 적분으로 바꾸기 — 고교 수준에서 출발하는 설명

## 0. 어떤 문제인가

표면 위의 한 점 $\mathbf{x}$에 여러 방향에서 빛이 들어온다. 이 점이 (Lambert 표면일 때) 얼마나 밝게 보이는지는

$$
\frac{1}{\pi}\int_{\Omega} L_i(\mathbf{x},\omega_i)\,(\omega_i\cdot\mathbf{n})\,d\omega_i
$$

로 쓴다. 여기서

- $\Omega$ : 표면의 법선 $\mathbf{n}$ 위쪽 **반구**(hemisphere). 표면 아래에서 오는 빛은 없으므로 반구만 본다.
- $\omega_i$ : 빛이 들어오는 방향(길이 1인 벡터).
- $L_i$ : 그 방향에서 들어오는 빛의 세기(radiance).
- $\omega_i\cdot\mathbf{n}$ : 입사 방향과 법선의 내적. 비스듬히 들어오는 빛은 덜 기여한다는 뜻(Lambert 코사인 법칙).
- $d\omega_i$ : "방향의 조각" — 구면 위의 아주 작은 면적(입체각, solid angle).

문제는 "$\int_\Omega \cdots d\omega$"가 컴퓨터가 직접 계산하기엔 너무 추상적이라는 점이다. 그래서 **각도 두 개**로 방향을 표시하는 구면 좌표계로 옮겨서, 고교에서 배운 평범한 정적분 두 겹으로 바꾼다.

## 1. 방향을 두 개의 각으로 표시하기 (구면 좌표계)

법선 $\mathbf{n}$을 $z$축이라 두자. 길이 1인 방향 벡터 $\omega$는 두 각으로 정해진다.

- $\theta$ (천정각, zenith angle): $z$축에서 얼마나 기울었는가. $z$축 방향이면 $0$, 수평이면 $\frac{\pi}{2}$, 정반대이면 $\pi$.
- $\phi$ (방위각, azimuth): $xy$평면에서 $x$축을 기준으로 얼마나 돌았는가. $0 \le \phi < 2\pi$.

고교 기하의 삼각비를 그대로 쓰면

$$
\omega = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta).
$$

확인: $z$성분은 $\cos\theta$, $xy$평면에 내린 그림자의 길이는 $\sin\theta$이고, 그 그림자를 다시 $\phi$로 나누면 $x=\sin\theta\cos\phi,\ y=\sin\theta\sin\phi$. 길이는 $\sin^2\theta(\cos^2\phi+\sin^2\phi)+\cos^2\theta = 1$로 잘 맞는다.

### 내적 항이 $\cos\theta$로 바뀌는 이유

$\mathbf{n} = (0,0,1)$이므로

$$
\omega_i\cdot\mathbf{n} = \cos\theta .
$$

즉 피적분 함수 속 $(\omega_i\cdot\mathbf{n})$은 그냥 $\cos\theta$다.

### 적분 범위

- 반구는 "법선 위쪽" $= z \ge 0$ $= \cos\theta \ge 0$ $= 0 \le \theta \le \frac{\pi}{2}$.
- 방위각은 한 바퀴 전부: $0 \le \phi \le 2\pi$.

**$\theta$의 상한이 $\frac{\pi}{2}$인 것이 바로 "반구"라는 뜻이다.** 구 전체라면 $\theta$가 $\pi$까지 간다.

## 2. $d\omega = \sin\theta\,d\theta\,d\phi$ — 왜 $\sin\theta$가 붙는가

여기가 핵심이다. $d\omega$는 구면 위의 작은 면적 조각인데, $\theta$를 $d\theta$만큼, $\phi$를 $d\phi$만큼 바꿨을 때 구면 위에서 실제로 훑는 면적이 얼마인지 구해야 한다.

반지름 1인 구면에서 각 $\theta$, $\phi$ 근처의 작은 조각을 보자.

- $\theta$ 방향으로 $d\theta$만큼 움직이면 구면 위 거리(호의 길이)는 반지름 $1$ × 각 $d\theta$ = $d\theta$.
- $\phi$ 방향으로 $d\phi$만큼 움직이면? $\phi$는 $z$축을 중심으로 도는 각이다. 이때 도는 원은 위도선(latitude circle)인데, 그 **반지름은 1이 아니라 $\sin\theta$**다 (위 좌표식에서 $xy$평면 그림자 길이가 $\sin\theta$였다). 그러므로 호의 길이는 $\sin\theta\,d\phi$.

두 변의 길이가 $d\theta$와 $\sin\theta\,d\phi$인 아주 작은 직사각형이므로

$$
d\omega = \sin\theta\,d\theta\,d\phi .
$$

직관: 지구 위에서 경도 1도 차이는 적도($\theta=\frac{\pi}{2}$, $\sin\theta=1$)에서는 약 111 km이지만 극 근처($\theta\to0$, $\sin\theta\to0$)에서는 거의 0 km다. 극 근처의 격자 칸은 실제로는 아주 작다. $\sin\theta$가 이 "칸 크기의 왜곡"을 보정한다.

### 검산: 반구의 넓이

$L=1$, 코사인 항도 빼고 그냥 $d\omega$만 반구에서 적분하면 반구의 표면적 $2\pi$가 나와야 한다.

$$
\int_0^{2\pi}\!\!\int_0^{\pi/2}\sin\theta\,d\theta\,d\phi
= 2\pi\,\bigl[-\cos\theta\bigr]_0^{\pi/2}
= 2\pi(0-(-1)) = 2\pi. \quad\checkmark
$$

$\sin\theta$가 없다면 $2\pi\cdot\frac{\pi}{2} = \pi^2 \approx 9.87 \ne 6.28$로 틀린다.

## 3. 조립하기

세 가지를 대입한다.

| 원래 식의 조각 | 구면 좌표에서 |
|---|---|
| $\Omega$ (반구) | $\phi\in[0,2\pi],\ \theta\in[0,\frac{\pi}{2}]$ |
| $\omega_i\cdot\mathbf{n}$ | $\cos\theta$ |
| $d\omega_i$ | $\sin\theta\,d\theta\,d\phi$ |

따라서

$$
\frac{1}{\pi}\int_{\Omega} L_i(\omega_i)(\omega_i\cdot\mathbf{n})\,d\omega_i
\;=\;
\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\frac{\pi}{2}} L(\theta,\phi)\,\cos\theta\,\sin\theta\;d\theta\,d\phi .
$$

$\cos\theta$는 **Lambert 법칙**(비스듬한 빛은 약하게), $\sin\theta$는 **좌표 변환의 면적 보정**(넓이 조각 크기)이라는 서로 다른 출처에서 왔다는 점을 기억하면 헷갈리지 않는다.

## 4. $\frac{1}{\pi}$의 의미 — 검산 겸 직관

모든 방향에서 똑같은 세기 $L$의 빛이 올 때(균일한 흐린 하늘) 결과는 얼마일까?

$$
\frac{L}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,d\phi
= \frac{L}{\pi}\cdot 2\pi\cdot\Bigl[\tfrac{1}{2}\sin^2\theta\Bigr]_0^{\pi/2}
= \frac{L}{\pi}\cdot 2\pi\cdot\tfrac12 = L .
$$

($\int\cos\theta\sin\theta\,d\theta$는 $u=\sin\theta$로 치환하는 고교 적분.) 즉 반구 전체가 밝기 $L$이면 표면도 정확히 $L$로 보인다 — 에너지가 새거나 늘어나지 않는다. Lambert BRDF에 $\frac{1}{\pi}$가 붙는 이유가 정확히 이 정규화다.

## 5. 컴퓨터가 계산하는 방식 (리만합)

원문은 이 적분을 그대로 셰이더에서 계산하기 위해 $\phi$를 $n_1$칸, $\theta$를 $n_2$칸으로 나눠 직사각형 넓이의 합(리만합)으로 근사한다.

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\cos\theta\sin\theta
= \frac{\pi}{n_1 n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\cos\theta\sin\theta .
$$

$\frac{2\pi}{n_1}$과 $\frac{\pi/2}{n_2}$는 각각 $d\phi$, $d\theta$에 해당하는 칸의 폭이다. 셰이더 코드의 `for theta < 0.5*PI`와 `* cos(theta) * sin(theta)`가 이 식을 그대로 옮긴 것이다.

## 한 줄 요약

반구 $\Omega$ → $\theta\in[0,\frac{\pi}{2}],\ \phi\in[0,2\pi]$; 내적 → $\cos\theta$; 입체각 조각 → $\sin\theta\,d\theta\,d\phi$. 합치면 $\frac{1}{\pi}\int_0^{2\pi}\!\int_0^{\pi/2} L\cos\theta\sin\theta\,d\theta\,d\phi$ 이고, 상한 $\frac{\pi}{2}$가 "반구"를 뜻한다.
