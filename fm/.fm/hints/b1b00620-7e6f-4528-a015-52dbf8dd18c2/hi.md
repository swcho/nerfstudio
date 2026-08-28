# Radiance의 구면 조화 함수 계수 $L_{lm}$은 어떻게 구하는가?

## 한 줄 답

$$
L_{lm}=\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi
$$

즉, **Radiance 함수 $L$에 기저 함수 $y_l^m$을 곱해 구 전체에서 적분**한다. 이 과정을 **투영(projection)**이라 부른다.

---

## 1. 벡터의 성분 구하기 — 출발점

고교 기하에서 배운 것부터 시작하자. 평면 벡터 $\vec v$가 있고, 서로 수직인 단위 벡터 $\hat e_1,\hat e_2$가 있으면

$$
\vec v = c_1\hat e_1 + c_2\hat e_2,\qquad c_i = \vec v\cdot\hat e_i
$$

각 축의 **계수(성분)**는 그 축 방향 단위 벡터와의 **내적**으로 구한다. 이것이 "투영"의 원형이다. $\hat e_i$들이 서로 수직이고 길이가 1(정규직교)이기 때문에 이렇게 간단히 나온다.

## 2. 함수도 벡터처럼 — 내적을 적분으로

함수 $f(x)$를 "성분이 무한히 많은 벡터"로 생각하자. 두 함수의 내적은 성분끼리 곱해서 더하는 것, 즉 **곱해서 적분**하는 것으로 정의한다.

$$
\langle f, g\rangle = \int f(x)\,g(x)\,dx
$$

기저 함수 $b_1(x), b_2(x), \dots$가 정규직교($\langle b_i,b_j\rangle = 0\ (i\ne j)$, $\langle b_i,b_i\rangle=1$)이면, 벡터와 완전히 같은 논리로

$$
f(x)=\sum_i c_i\,b_i(x),\qquad c_i=\langle f,b_i\rangle=\int f(x)\,b_i(x)\,dx
$$

원문(투영 절)이 말하는 "원본 함수의 전체 영역에 걸쳐 기저 함수와의 곱을 적분하면 계수가 나온다"가 바로 이 식이다.

## 3. 구 위의 함수라면 — 적분 영역이 구 표면

Radiance $L$은 **방향**마다 값이 있는 함수다. 방향은 구 표면의 한 점이므로 구면 좌표 $(\theta,\phi)$로 쓴다($\theta$: 천정각 $0\sim\pi$, $\phi$: 방위각 $0\sim2\pi$). 기저 함수는 구 위에서 정규직교인 **구면 조화 함수 $y_l^m(\theta,\phi)$**를 쓴다.

그러면 2절의 식에서 $\int\cdots dx$를 "구 표면 전체에 대한 적분"으로 바꾸면 끝이다.

$$
L_{lm}=\int_{\text{구 전체}} L\,y_l^m\;d\omega
$$

## 4. $\sin\theta$는 왜 붙는가?

구 표면의 아주 작은 면적 조각 $d\omega$를 $(\theta,\phi)$로 표현하면 $d\omega=\sin\theta\,d\theta\,d\phi$ 이다.

- 위도 방향으로 $d\theta$만큼 움직이면 호 길이는 $d\theta$ (반지름 1).
- 경도 방향으로 $d\phi$만큼 움직이면 호 길이는 $\sin\theta\,d\phi$ — 위도선 원의 반지름이 $\sin\theta$이기 때문이다(극 근처는 원이 작고, 적도는 크다).

두 길이를 곱한 넓이가 $\sin\theta\,d\theta\,d\phi$. 그래서 최종 식이

$$
L_{lm}=\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi
$$

가 된다. 확인: $L=y_l^m=1$ 처럼 놓고 적분하면 $\int_0^{2\pi}\!\int_0^\pi \sin\theta\,d\theta\,d\phi = 2\pi\cdot 2 = 4\pi$, 즉 단위구의 표면적이 나온다.

## 5. 왜 반구가 아니라 "구 전체"인가?

Irradiance 원식은 표면 위쪽 **반구**($\theta\le\pi/2$)에서 $L\cos\theta$를 적분한다. 원문은 이를 두 조각으로 나눈다.

1. $L$ 자체는 **구 전체**에서 투영해 $L_{lm}$을 얻는다 (지금 카드).
2. $\max(\cos\theta,0)$을 따로 투영해 $A_l$을 얻는다. 이 함수가 아래 반구에서 0이므로, 둘을 곱하면 자동으로 반구만 남는다.

이렇게 나눠야 조명(환경맵) 부분 $L_{lm}$을 표면 방향과 무관하게 **한 번만** 계산해 두고 재사용할 수 있다. 그래서 $L_{lm}$의 적분 범위는 $\theta: 0\to\pi$, 즉 구 전체다.

## 6. 실제 계산 — 적분을 합으로

컴퓨트 셰이더에서는 적분을 리만 합(구분구적법)으로 바꾼다. $\phi$를 $n_1$칸, $\theta$를 $n_2$칸으로 나누면

$$
L_{lm}\approx\frac{2\pi}{n_1}\cdot\frac{\pi}{n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta
$$

각 방향에서 큐브맵을 샘플링해 $L$을 읽고, $y_l^m$과 $\sin\theta$를 곱해 누적하면 된다. $l\le2$면 $(l,m)$ 조합이 9개이므로 RGB 각각 9개, 총 27개 숫자로 조명 전체를 요약한다.

## 정리

| 단계 | 벡터 | 함수(구 위) |
|---|---|---|
| 기저 | 정규직교 단위벡터 $\hat e_i$ | 구면 조화 함수 $y_l^m$ |
| 계수 구하기 | $c_i=\vec v\cdot\hat e_i$ | $L_{lm}=\int L\,y_l^m\,\sin\theta\,d\theta\,d\phi$ |
| 복원 | $\vec v=\sum c_i\hat e_i$ | $L\approx\sum L_{lm}\,y_l^m$ |

**핵심 기억법**: "계수 = 함수 × 기저를 구 전체에서 적분", 그리고 구면 적분에는 항상 $\sin\theta$가 따라온다.
