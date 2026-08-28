# 리만합으로 바꾼 픽셀 셰이더용 최종 계산식 — 고교 수학으로 따라가기

## 1. 무엇을 계산하려는가

Irradiance Map은 "어느 방향 $\mathbf{n}$을 바라보는 표면이 주변(큐브맵 하늘)에서 받는 빛의 총량"을 미리 계산해 두는 것입니다. 원문은 알베도 $\sigma$를 뺀 Lambert 확산 반사식에서 출발합니다.

$$
\frac{1}{\pi}\int_{\Omega} L_i(\mathbf{x},\omega_i)\,(\omega_i\cdot\mathbf{n})\,d\omega_i
$$

- $\Omega$ : 법선 $\mathbf{n}$ 위쪽의 **반구**(하늘 절반)
- $L_i$ : 방향 $\omega_i$에서 들어오는 빛의 밝기 (큐브맵에서 읽음)
- $\omega_i\cdot\mathbf{n}$ : 빛이 비스듬히 들어오면 약해지는 효과. 두 단위벡터의 내적이므로 $\cos\theta$
- $\frac{1}{\pi}$ : Lambert BRDF 상수 (에너지 보존을 위해 붙는 값)

## 2. 반구 위의 적분을 구면 좌표로 쓰기

고교 기하에서 배운 구면 좌표를 씁니다. 법선을 $z$축으로 두면 방향 하나는 두 각으로 정해집니다.

- $\theta$ (천정각): 법선에서 벗어난 각도, 반구이므로 $0\le\theta\le\frac{\pi}{2}$
- $\phi$ (방위각): 법선 둘레를 한 바퀴 도는 각도, $0\le\phi\le 2\pi$

핵심은 "구 표면의 작은 조각 넓이" $d\omega$입니다. 단위구에서 위도 방향으로 $d\theta$, 경도 방향으로 $d\phi$만큼 움직이면, 경도 방향 조각의 실제 길이는 반지름 $\sin\theta$짜리 원 위에 있으므로 $\sin\theta\,d\phi$입니다. 따라서

$$
d\omega = \sin\theta\,d\theta\,d\phi .
$$

(극 근처($\theta\approx0$)에서는 조각이 작고, 적도 근처($\theta\approx\frac{\pi}{2}$)에서는 크다는 뜻입니다. 지구본에서 경도선 간격이 극으로 갈수록 좁아지는 것과 같습니다.)

$\omega_i\cdot\mathbf{n}=\cos\theta$를 넣으면 이중적분이 됩니다.

$$
\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L(\theta,\phi)\,\cos\theta\,\sin\theta\;d\theta\,d\phi
$$

## 3. 리만합: 적분을 "칸 나눠서 더하기"로 바꾸기

고교 미적분에서 정적분의 정의는 리만합의 극한이었습니다. 구간 $[a,b]$를 $n$칸으로 나누면 한 칸의 폭은 $\Delta x=\frac{b-a}{n}$이고

$$
\int_a^b f(x)\,dx \approx \sum_{k} f(x_k)\,\Delta x .
$$

GPU의 픽셀 셰이더는 적분 기호를 모르고 **반복문으로 더하기**만 할 수 있으므로, 이 근사식이 그대로 실행 가능한 형태입니다.

이중적분이라 칸을 두 방향으로 나눕니다.

- $\phi$ 방향: 구간 길이 $2\pi$를 $n_1$칸 → $\Delta\phi=\dfrac{2\pi}{n_1}$
- $\theta$ 방향: 구간 길이 $\dfrac{\pi}{2}$를 $n_2$칸 → $\Delta\theta=\dfrac{\pi/2}{n_2}$

$d\theta\,d\phi$를 $\Delta\theta\,\Delta\phi$로 바꾸고 상수는 합 밖으로 꺼내면

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2}\;\sum_{\phi=0}^{n_1}\sum_{\theta=0}^{n_2} L_i(\mathbf{x},\omega_i)\cos\theta\,\sin\theta .
$$

## 4. 상수 정리 — 답의 $\frac{\pi}{n_1 n_2}$가 나오는 과정

앞의 세 상수를 곱하기만 하면 됩니다.

$$
\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi}{2n_2}
=\frac{1\cdot 2\pi\cdot\pi}{\pi\cdot n_1\cdot 2n_2}
=\frac{2\pi^2}{2\pi\,n_1 n_2}
=\frac{\pi}{n_1 n_2}.
$$

분자의 $2$와 분모의 $2$, 분자의 $\pi$ 하나와 분모의 $\pi$가 약분되어 $\pi$ 하나만 남습니다. 그래서 최종식은

$$
\boxed{\;\frac{\pi}{n_1 n_2}\sum_{\phi=0}^{n_1}\sum_{\theta=0}^{n_2} L_i(\mathbf{x},\omega_i)\cos\theta\,\sin\theta\;}
$$

## 5. 셰이더 코드와 1:1 대응

원문 HLSL은 이 식을 그대로 옮긴 것입니다.

```
for phi in [0, 2π) step Δ:
    for theta in [0, π/2) step Δ:
        irradiance += CubeMap(방향) * cos(theta) * sin(theta)   // 합 안의 항
        ++numSample                                            // n1·n2 세기
irradiance = PI * irradiance / numSample                       // π / (n1 n2) 곱하기
```

`numSample`이 두 반복문의 총 반복 수, 즉 $n_1 n_2$이고, 앞에 곱한 `PI`가 약분 후 남은 $\pi$입니다.

## 6. 간단한 검산

하늘이 모든 방향에서 밝기 $1$로 균일하다면($L\equiv1$) 실제 적분값은

$$
\frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,d\phi
=\frac{1}{\pi}\cdot2\pi\cdot\frac{1}{2}=1 .
$$

($\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta=\left[\tfrac{1}{2}\sin^2\theta\right]_0^{\pi/2}=\tfrac12$ — 고교 치환적분)

즉 "밝기 1인 하늘 아래 흰 표면은 밝기 1"이라는 직관과 맞고, 리만합도 $n_1,n_2$를 키우면 1에 가까워져야 합니다. 이것이 $\frac{\pi}{n_1n_2}$라는 계수가 올바른지 확인하는 가장 쉬운 방법입니다.
