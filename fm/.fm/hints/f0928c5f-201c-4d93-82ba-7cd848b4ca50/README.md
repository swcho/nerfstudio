# $A_l$의 정의와 $m=0$인 이유

## 카드 요약

- **정의**: $A_l=\displaystyle\int_{\theta=0}^{\pi}\max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta$
  — "클램프된 코사인" 함수 $\max(\cos\theta,0)$를 구면 조화 함수 기저 $y_l^0$에 **투영(projection)** 해서 얻는 계수.
- **$m=0$인 이유**: $\max(\cos\theta,0)$는 천정각 $\theta$에만 의존하고 방위각 $\phi$와 무관하다. $\phi$ 방향 변화를 담는 $m\ne 0$ 항의 계수는 전부 0이 되고, $\phi$에 독립인 $m=0$(zonal harmonics) 항만 남는다.

## 어디서 나온 식인가 — Irradiance 적분의 분해

Lambertian 표면의 Irradiance는 법선 방향을 $\theta=0$으로 놓은 로컬 좌표에서

$$
E=\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L_i(\omega_i)\cos\theta\,\sin\theta\,d\theta\,d\phi
$$

이다. 원문은 이 적분을 두 조각으로 나눈다.

1. **Radiance $L(\theta,\phi)$** — 반구 적분을 구 전체($\theta\in[0,\pi]$) 적분으로 확장.
2. **$\max(\cos\theta,0)$** — 반구 뒤쪽($\theta>\pi/2$)에서 0이 되도록 코사인을 잘라낸 함수. 구 전체에 대한 Radiance 적분과 곱해지면 자동으로 반구만 남는다.

각각을 SH 기저에 투영하면 두 계수 집합이 생긴다.

$$
L_{lm}=\int_{0}^{2\pi}\!\!\int_{0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi,
\qquad
A_l=\int_{0}^{\pi}\max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta
$$

$L_{lm}$은 환경맵(빛)의 계수, $A_l$은 **재질(Lambertian 코사인 항)** 의 계수다. 빛은 장면마다 바뀌지만 $A_l$은 항상 같은 함수의 투영이므로 **한 번 계산해 상수로 박아 둘 수 있다** — 이것이 SH 조명이 빠른 핵심 이유다.

## 왜 $m$은 항상 0인가

구면 조화 함수는 $y_l^m(\theta,\phi)=K_l^m\,P_l^{|m|}(\cos\theta)\times\{\cos(m\phi)\ \text{또는}\ \sin(|m|\phi)\}$ 꼴이다. $m\ne0$이면 $\phi$에 대해 $\cos(m\phi)$나 $\sin(m\phi)$가 붙는다.

투영 계수를 구할 때 $\phi$ 적분을 먼저 하면

$$
\int_0^{2\pi}\underbrace{\max(\cos\theta,0)}_{\phi\text{와 무관}}\cdot\cos(m\phi)\,d\phi
=\max(\cos\theta,0)\int_0^{2\pi}\cos(m\phi)\,d\phi = 0\quad(m\neq 0)
$$

가 된다. 한 주기(또는 정수 배 주기) 동안 $\cos(m\phi)$, $\sin(m\phi)$의 적분은 0이기 때문이다. 즉 **회전 대칭(방위각 불변)인 함수는 zonal harmonics($m=0$)만으로 완전히 표현된다.** 그래서 $A$에는 첨자가 $l$ 하나만 남고, 기저도 $y_l^0(\theta,0)$처럼 $\phi$를 아무 값(0)으로 고정해 쓴다.

물리적으로도 당연하다: 법선 방향을 축으로 표면을 빙글 돌려도 $\cos\theta$ 가중치는 변하지 않으므로, 그 가중치에 방위각 성분이 있을 수 없다.

## 실제 값과 감쇠 (Ramamoorthi & Hanrahan)

레퍼런스 논문(Ramamoorthi & Hanrahan, *On the relationship between radiance and irradiance*, 2001)은 $A_l$을 닫힌 형태로 구했다.

- $A_0=\dfrac{\sqrt{\pi}}{2}\approx 0.886$
- $A_1=\sqrt{\dfrac{\pi}{3}}\approx 1.023$
- $A_2=\dfrac{\sqrt{5\pi}}{8}\approx 0.495$
- $l\ge 3$이고 **홀수**이면 $A_l=0$
- $l$ 짝수이면 $A_l=2\pi\sqrt{\dfrac{2l+1}{4\pi}}\,\dfrac{(-1)^{l/2-1}}{(l+2)(l-1)}\left[\dfrac{l!}{2^l\,(l/2)!^2}\right]$ → $A_4\approx-0.111$, $A_6\approx 0.050$, 이후 $\sim l^{-2}$로 감소

![$l$에 따른 Lambertian BRDF 계수 $A_l$ (Ramamoorthi & Hanrahan)](fig-1.png)

그림에서 확인할 점:
- 가로축 $l$, 세로축 $A_l$. $l=0$에서 약 0.89, $l=1$에서 최대값 약 1.02, $l=2$에서 약 0.5.
- $l=3$에서 정확히 0으로 떨어지고, $l=4$에서 약 $-0.11$로 잠깐 음수, $l=5$ 다시 0, $l=6$ 약 $0.05$ … 이후 0 주위에서 미세하게 진동하며 사라진다. **홀수 $l\ge3$에서 모두 0**이라는 성질이 점선 위의 점들로 보인다.
- $l\le 2$ 세 항의 크기가 압도적이어서, $l\ge3$을 버려도 오차가 1% 수준이다. 이것이 "SH 계수 9개 × RGB = 27개 상수"로 Irradiance Map을 근사할 수 있는 근거다.

## $A_l$이 Irradiance 복원에 쓰이는 방식

$$
E_{lm}=\sqrt{\frac{4\pi}{2l+1}}\,A_l\,L_{lm}=\hat A_l\,L_{lm},
\qquad
E(\theta,\phi)=\sum_{l,m}E_{lm}\,y_l^m(\theta,\phi)
$$

$\sqrt{4\pi/(2l+1)}$는 로컬 좌표(법선 = $z$축)에서 계산한 $A_l$을 월드 좌표의 임의 법선으로 **회전**시킬 때 붙는 가중치다(구면 조화 함수의 회전은 같은 $l$ 안에서만 섞이므로 계수가 $l$에만 의존한다). 원문이 미리 계산해 둔 상수는

$$
\hat A_0=\pi,\quad \hat A_1=\tfrac{2\pi}{3}\approx2.094,\quad \hat A_2=\tfrac{\pi}{4}\approx0.785,\quad \hat A_3=0,\quad \hat A_4\approx-0.131,\quad \hat A_5=0,\quad \hat A_6\approx0.049
$$

이다. 결국 Irradiance는 "환경맵 계수 $L_{lm}$에 $l$별 상수 $\hat A_l$을 곱한 것"이며, $\max(\cos\theta,0)$과의 컨볼루션이 주파수 영역에서 단순 곱셈으로 바뀐 셈이다.

## 표기에 대한 주의

원문 식에는 구면 측도 $\sin\theta$와 $\phi$ 적분에서 나오는 $2\pi$가 생략되어 있다. Ramamoorthi & Hanrahan의 원식은

$$
A_l=2\pi\int_0^{\pi/2}\cos\theta\;Y_{l0}(\theta)\,\sin\theta\,d\theta
$$

이며, $\max(\cos\theta,0)$을 쓰면 적분 범위를 $[0,\pi]$로 늘려도 같은 값이다. 위에 적은 수치들은 이 완전한 식으로 계산한 것이다. 카드에서 기억할 핵심은 "클램프된 코사인을 $y_l^0$에 투영한 계수"라는 구조와 "$\phi$ 독립 → $m=0$"이라는 논리다.
