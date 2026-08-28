# Irradiance의 SH 계수 $E_{lm}$과 $L_{lm}$, $A_l$의 관계

## 한 줄 답

$$E_{lm}=\sqrt{\frac{4\pi}{2l+1}}\,A_l\,L_{lm}$$

Irradiance는 "입사 radiance $L$"과 "clamped cosine $\max(\cos\theta,0)$"의 **구면 컨볼루션**이다. 구면 조화(SH) 기저에서 컨볼루션은 계수끼리의 곱으로 바뀌므로, irradiance의 SH 계수 $E_{lm}$은 radiance의 계수 $L_{lm}$에 cosine 커널의 계수 $A_l$을 곱한 것이 된다. 앞에 붙는 $\sqrt{4\pi/(2l+1)}$은 로컬 좌표(법선을 $z$축으로 둔 좌표)에서 구한 $\cos\theta$의 계수를 월드 좌표의 법선 방향으로 **회전**시킬 때 생기는 정규화 가중치다.

## 어디서 나오는 식인가

원문(spherical-harmonics.md)은 Lambert diffuse의 irradiance를

$$E(\mathbf n)=\int_{\Omega(\mathbf n)} L_i(\omega)\,(\omega\cdot\mathbf n)\,d\omega
=\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L_i(\theta,\phi)\cos\theta\,\sin\theta\,d\theta\,d\phi$$

로 쓰고, 이것을 두 조각으로 분리한다.

1. **radiance 부분** — 반구 적분을 구 전체 적분으로 확장한 $L(\theta,\phi)$.
2. **cosine 부분** — $\max(\cos\theta,0)$. 반구 바깥에서는 0이 되므로 곱하면 다시 반구만 남는다.

각각을 SH 기저 $y_l^m$에 투영하면

$$L_{lm}=\int_{\phi}\int_{\theta} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi,\qquad
A_l=\int_{\theta} \max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta$$

![투영: 원함수 × 기저함수를 적분하면 계수 c_i가 나온다](fig-2.png)

위 그림(원문 img-5, Green 2003)은 이 "투영"의 뜻을 1차원으로 보여준다. 왼쪽의 원래 함수와 오른쪽의 각 기저 함수를 곱해 적분하면 계수 $c_1, c_2, c_3$이 하나씩 나온다. $L_{lm}$은 큐브맵 radiance를 $y_l^m$과 곱해 적분한 것, $A_l$은 $\max(\cos\theta,0)$를 $y_l^0$과 곱해 적분한 것 — 정확히 같은 연산이다.

$\cos\theta$는 천정각 $\theta$에만 의존하고 방위각 $\phi$에는 무관하므로(원형 대칭, zonal), $m\neq 0$인 계수는 모두 0이 되어 $A_l$은 첨자가 $l$ 하나뿐이다.

## 왜 곱이 되고, 왜 $\sqrt{4\pi/(2l+1)}$이 붙는가

$E(\mathbf n)$을 SH로 전개하면 $E(\mathbf n)=\sum_{l,m}E_{lm}\,y_l^m(\mathbf n)$이고, Ramamoorthi & Hanrahan(JOSA 2001, 원문 레퍼런스 4의 4.A)의 유도에 따르면

$$E_{lm}=\hat A_l\,L_{lm},\qquad \hat A_l=\sqrt{\frac{4\pi}{2l+1}}\,A_l.$$

핵심 논리는 다음과 같다.

- $A_l$은 **법선이 $+z$축**인 로컬 좌표에서 계산한 cosine 커널의 계수다. 실제 표면의 법선 $\mathbf n$은 임의의 방향이므로 커널을 $\mathbf n$ 방향으로 회전시켜야 한다.
- SH는 회전에 대해 "같은 $l$ 안에서 닫혀" 있고, zonal 함수($m=0$)를 방향 $\mathbf n$으로 회전한 결과는 **Funk–Hecke 정리(구면 컨볼루션 정리)** 에 의해 $y_l^m(\mathbf n)$들의 선형 조합으로 표현된다. 그때 나오는 정규화 상수가 $\sqrt{4\pi/(2l+1)}$이다. 이 값은 $y_l^0$의 정규화 상수 $\sqrt{(2l+1)/4\pi}$의 역수로, 회전된 커널을 다시 orthonormal SH 계수로 표현할 때 나타난다.
- 결과적으로 $E(\mathbf n)=\sum_{l,m}\hat A_l L_{lm}\,y_l^m(\mathbf n)$, 즉 "radiance 계수 × 커널 계수(회전 가중치 포함)"가 된다. 컨볼루션이 곱으로 바뀌는 Fourier 컨볼루션 정리의 구면 버전이라고 보면 된다.

## $\hat A_l$의 실제 값과 그 의미

원문이 제시하는 미리 계산된 상수는

$$\hat A_0=3.1415,\ \hat A_1=2.0943,\ \hat A_2=0.7853,\ \hat A_3=0,\ \hat A_4=-0.1309,\ \hat A_5=0,\ \hat A_6=0.0490$$

즉 $\hat A_0=\pi$, $\hat A_1=2\pi/3$, $\hat A_2=\pi/4$이고, $l\ge 3$인 홀수 차수는 모두 0, 짝수 차수는 $l^{-5/2}$ 정도로 빠르게 줄어든다.

![Lambertian BRDF coefficient vs l — l=3부터 거의 0](fig-1.png)

위 그래프(원문 img-8, Ramamoorthi & Hanrahan)는 $\sqrt{4\pi/(2l+1)}$을 곱하기 **전**의 $A_l$을 $l=0\ldots20$에 대해 찍은 것이다. $l=0$에서 약 0.886($=\sqrt{\pi}/2$), $l=1$에서 최대 약 1.02, $l=2$에서 약 0.495, $l=3$에서 정확히 0으로 떨어진 뒤 $l=4$에서 약 $-0.11$로 살짝 음수가 되고, 이후 점선(0)을 따라 미세하게 진동하며 소멸한다. 홀수 $l\ge3$에서 점이 모두 점선 위에 놓여 있는 것을 확인할 수 있다. 여기에 $\sqrt{4\pi/(2l+1)}$을 곱하면 $l=0$: $0.886\times3.545=3.14=\pi$, $l=1$: $1.02\times2.047=2.09$, $l=2$: $0.495\times1.585=0.785$로 원문 표의 $\hat A_l$이 정확히 재현된다.

이 급격한 감쇠가 **$l\le2$까지 9개 계수(RGB면 27개)** 만으로 irradiance map을 근사해도 오차가 1% 수준에 그치는 이유다. 32×32×6 큐브맵 24 KB 대신 108 byte로 거의 같은 결과를 얻는 원문의 최적화가 여기서 정당화된다.

## 셰이더 코드와의 연결

원문의 `ImageBasedLight()`에 등장하는 상수 $c_1\ldots c_5$는 $\hat A_l$과 $y_l^m$의 정규화 상수를 미리 곱해 둔 값이다. 예컨대

- $c_4=0.886227=\hat A_0\cdot Y_{00}=\pi\cdot\frac{1}{2\sqrt\pi}$
- $c_2=0.511664=\hat A_1\cdot\frac{1}{2}\sqrt{\frac{3}{\pi}}$
- $c_1=0.429043=\hat A_2\cdot\frac{1}{4}\sqrt{\frac{15}{\pi}}$

따라서 셰이더가 GPU에 올려 두는 값은 $L_{lm}$ 27개뿐이고, $E_{lm}=\hat A_l L_{lm}$의 $\hat A_l$ 곱셈은 상수 $c_i$ 안에 접혀 들어가 매 픽셀마다 `normal`의 다항식으로 평가된다.

## 요약

| 기호 | 의미 | 어디서 오나 |
|---|---|---|
| $L_{lm}$ | 입사 radiance(큐브맵)의 SH 계수 | 큐브맵을 $y_l^m$에 투영 |
| $A_l$ | $\max(\cos\theta,0)$의 zonal SH 계수 ($m=0$) | 로컬 좌표($\mathbf n=+z$)에서 투영 |
| $\sqrt{4\pi/(2l+1)}$ | 로컬 → 월드 회전 정규화 가중치 | Funk–Hecke / SH 회전 성질 |
| $\hat A_l$ | $\sqrt{4\pi/(2l+1)}A_l$, 미리 계산된 상수 | $\pi,\ 2\pi/3,\ \pi/4,\ 0,\ -\pi/24,\ 0,\ \ldots$ |
| $E_{lm}$ | irradiance의 SH 계수 | $\hat A_l L_{lm}$ (컨볼루션 = 계수 곱) |

## 참고

- 원문: `spherical-harmonics.md` — "구면 조화 함수를 이용한 Irradiance Map" 절
- Ramamoorthi, Hanrahan. *On the relationship between radiance and irradiance*, JOSA A 2001, §4.A (유도), Fig. 3 (fig-1)
- Ramamoorthi, Hanrahan. *An Efficient Representation for Irradiance Environment Maps*, SIGGRAPH 2001, §3.2 ($c_1\ldots c_5$)
- Green. *Spherical Harmonic Lighting: The Gritty Details*, 2003 (fig-2)
