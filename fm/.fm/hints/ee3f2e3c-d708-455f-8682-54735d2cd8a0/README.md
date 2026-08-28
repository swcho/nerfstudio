# $\hat{A_l}$의 정의와 도입 이유

## 한 줄 요약

$$\hat{A_l}=\sqrt{\frac{4\pi}{2l+1}}\,A_l$$

Irradiance의 SH 계수를 구하는 식 $E_{lm}=\sqrt{\frac{4\pi}{2l+1}}A_l\,L_{lm}$에서, **$L_{lm}$을 제외한 나머지 전부**(회전 가중치 × 클램프 코사인 계수)를 하나의 상수로 묶은 것이다. 이 값은 조명(환경맵)과 무관하게 $l$에만 의존하므로 **미리 계산해 두고 런타임에서는 곱셈 한 번만** 하면 된다.

## 어디에서 나오는가

Irradiance 적분

$$E(\mathbf n)=\int_{\Omega} L(\omega)\,\max(\cos\theta,0)\,d\omega$$

를 두 조각으로 나눠 각각 SH로 투영한다.

| 조각 | SH 계수 | 성질 |
|---|---|---|
| 환경 radiance $L(\theta,\phi)$ | $L_{lm}=\int L\,y_l^m\,d\omega$ | 씬(환경맵)마다 다름 → 런타임/베이크 시점에 계산 |
| 클램프 코사인 $\max(\cos\theta,0)$ | $A_l=\int_0^{\pi}\max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta$ | 천정각만 의존 → $m=0$, **조명과 무관한 순수 상수** |

SH 공간에서 두 함수의 컨볼루션은 계수끼리의 곱이 되지만(Funk–Hecke 정리), 로컬 좌표(법선을 $z$축으로 둔 좌표)에서 구한 $A_l$을 월드 좌표의 $L_{lm}$과 맞추려면 $\sqrt{\frac{4\pi}{2l+1}}$이라는 정규화 인자가 붙는다(Ramamoorthi & Hanrahan 2001, 4.A절). 그래서

$$E_{lm}=\sqrt{\frac{4\pi}{2l+1}}\,A_l\,L_{lm}\quad\Longrightarrow\quad E_{lm}=\hat{A_l}\,L_{lm}$$

이 된다. 즉 $\hat{A_l}$은 "**Lambertian 커널(클램프 코사인)의 SH 주파수 응답**"이라고 볼 수 있다: $l$번째 주파수 대역의 조명이 irradiance로 얼마나 전달되는지 나타내는 이득(gain)이다.

## 왜 도입하는가

1. **런타임 계산 제거** — $\sqrt{4\pi/(2l+1)}$과 $A_l$ 모두 조명·카메라·물체와 무관하게 $l$만의 함수다. 매 프레임/매 픽셀 다시 계산할 이유가 없으므로 상수 테이블로 굳혀 두고, 셰이더에서는 `E_lm = Ahat[l] * L_lm` 곱셈만 남긴다.
2. **수식 단순화** — 두 인자를 하나로 합쳐 $E_{lm}=\hat{A_l}L_{lm}$이라는 한눈에 들어오는 형태가 되고, 실제 구현에서는 이 값을 $L_{lm}$에 미리 곱해 둔 뒤 $y_l^m(\mathbf n)$과 내적하는 형태로 정리된다.
3. **대역 절단(band-limit)의 근거 제공** — 값이 급격히 감소하므로 어디까지 계산할지 결정하는 기준이 된다(아래).

## 미리 계산된 값

닫힌 형태로 구할 수 있다(Ramamoorthi & Hanrahan):

- $\hat{A_0}=\pi\approx 3.1416$
- $\hat{A_1}=\tfrac{2\pi}{3}\approx 2.0944$
- $\hat{A_2}=\tfrac{\pi}{4}\approx 0.7854$
- $\hat{A_3}=0$
- $\hat{A_4}=-\tfrac{\pi}{24}\approx -0.1309$
- $\hat{A_5}=0$
- $\hat{A_6}=\tfrac{\pi}{64}\approx 0.0491$

일반식: $l=1$은 $\tfrac{2\pi}{3}$, 홀수 $l>1$은 0, 짝수 $l$은 $2\pi\,\frac{(-1)^{l/2-1}}{(l+2)(l-1)}\Big[\frac{l!}{2^l(l/2!)^2}\Big]$.

![Lambertian BRDF(클램프 코사인)의 SH 계수 $A_l$, $l=0\ldots20$](fig-1.png)

그림(원 논문 Fig.)은 $\hat{A_l}$이 아닌 정규화 이전의 $A_l$을 그린 것이다. $l=0$에서 약 0.886($=\sqrt{\pi}/2$), $l=1$에서 약 1.02($=\sqrt{\pi/3}$), $l=2$에서 약 0.495로 시작해 $l=3$에서 정확히 0, $l=4$에서 약 $-0.11$의 작은 음수, 이후 홀수는 전부 0이고 짝수는 부호를 바꾸며 0으로 빠르게 수렴한다. 각 점에 $\sqrt{4\pi/(2l+1)}$을 곱하면 위의 $\hat{A_l}$ 표가 그대로 나온다(예: $0.886\times\sqrt{4\pi}=\pi$). 형태는 같으므로 **감쇠 경향을 보는 데는 어느 쪽을 봐도 동일**하다.

## 결과: $l\le 2$로 충분

$\hat{A_l}$이 $l=2$ 이후 급감하기 때문에(3차 이상은 0 또는 1% 수준) irradiance는 사실상 저주파 신호다. 따라서 $l\le 2$의 9개 SH 계수만 유지하면 되고, RGB 3채널이면 **27개 상수**로 환경 조명 전체의 diffuse 응답을 표현할 수 있다. 이것이 셰이더의 `ShFunctionL2`(9개 기저 함수만 평가)로 이어진다. 이 절단 오차는 평균적으로 약 1% 이하로 알려져 있다.

## 기억 포인트

- 정의: $\hat{A_l}=\sqrt{\frac{4\pi}{2l+1}}A_l$ — "회전 가중치 × 클램프 코사인 계수".
- 목적: 조명과 무관한 상수이므로 **사전 계산 → 런타임 곱셈 한 번**.
- 값: $\pi,\ \tfrac{2\pi}{3},\ \tfrac{\pi}{4},\ 0,\ -\tfrac{\pi}{24},\ 0,\ \tfrac{\pi}{64}$ — 급격히 줄어들어 $l\le2$ 절단을 정당화.

## 참고

- Ramamoorthi & Hanrahan, *On the relationship between radiance and irradiance: determining the illumination from images of a convex Lambertian object*, JOSA A 2001, 4.A절.
- Ramamoorthi & Hanrahan, *An Efficient Representation for Irradiance Environment Maps*, SIGGRAPH 2001.
