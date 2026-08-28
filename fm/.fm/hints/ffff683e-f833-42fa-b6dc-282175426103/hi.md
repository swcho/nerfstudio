# 정규화 배율 계수 $K_l^m$ — 고교 수준에서 쌓아 올리기

## 1. 먼저 결론

실수 구면 조화 함수 $y_l^m(\theta,\varphi)$ 안에 곱해져 있는 상수

$$
K_l^m=\sqrt{\frac{2l+1}{4\pi}\cdot\frac{(l-|m|)!}{(l+|m|)!}}
$$

는 **"기저 함수 하나의 크기(길이)를 1로 맞추는 배율"**입니다. 모양은 버금 르장드르 다항식 $P_l^m(\cos\theta)$와 $\cos(m\varphi)$, $\sin(m\varphi)$가 정하고, $K_l^m$은 오직 크기만 조절합니다.

## 2. "함수의 크기"란? — 벡터에서 출발

고교 기하에서 벡터 $\vec v=(a,b,c)$의 길이는 $|\vec v|=\sqrt{a^2+b^2+c^2}$이고, 두 벡터가 수직이면 내적 $\vec u\cdot\vec v=0$입니다. 길이가 1이면서 서로 수직인 벡터들의 모음($\hat i,\hat j,\hat k$)을 **정규 직교 기저**라고 부르죠.

함수도 똑같이 다룰 수 있습니다. 성분이 유한개가 아니라 "구면 위의 모든 점에서의 함수값"이므로, 덧셈 $\sum$ 대신 적분 $\int$을 씁니다.

- 두 함수의 내적: $\langle f,g\rangle=\displaystyle\int_{\text{구면}} f\,g\,dS$
- 함수의 크기(노름): $\|f\|=\sqrt{\langle f,f\rangle}=\sqrt{\displaystyle\int_{\text{구면}} f^2\,dS}$

여기서 $dS$는 구면 위 작은 넓이 조각으로, 구면좌표 $(\theta,\varphi)$로 쓰면 $dS=\sin\theta\,d\theta\,d\varphi$입니다. (반지름 1인 구의 전체 넓이가 $\int dS=4\pi$가 되는 것을 확인해 보세요 — 이 $4\pi$가 $K_l^m$의 분모에 있는 $4\pi$입니다.)

## 3. 정규 직교 기저가 왜 필요한가

구면 위의 밝기 분포 같은 함수 $f$를 기저 함수들의 합으로 근사한다고 합시다.

$$
f\approx\sum_{l,m} c_l^m\,y_l^m
$$

기저가 **정규 직교**($\langle y_l^m,y_{l'}^{m'}\rangle=0$ if $(l,m)\ne(l',m')$, $\|y_l^m\|=1$)이면, 계수는 벡터의 정사영처럼 그냥 내적 한 번으로 구해집니다.

$$
c_l^m=\langle f,\,y_l^m\rangle=\int f\,y_l^m\,dS
$$

크기가 1이 아니면 $c_l^m=\langle f,y_l^m\rangle/\|y_l^m\|^2$처럼 매번 나눠 줘야 하고, 함수마다 다른 스케일이 섞여 코드가 지저분해집니다. 그래서 처음부터 $\|y_l^m\|=1$이 되도록 $K_l^m$을 곱해 두는 것입니다.

## 4. $K_l^m$의 각 조각이 어디서 오는가

$y_l^m$은 $\theta$ 부분($P_l^m(\cos\theta)$)과 $\varphi$ 부분($\cos m\varphi$ 또는 $\sin m\varphi$)의 곱이므로, 크기의 제곱도 두 적분의 곱으로 쪼개집니다.

**(a) $\varphi$ 부분** — 고교 삼각함수 적분입니다.

$$
\int_0^{2\pi}\cos^2(m\varphi)\,d\varphi=\int_0^{2\pi}\frac{1+\cos 2m\varphi}{2}\,d\varphi=\pi\quad(m\neq 0)
$$

$m=0$이면 $\cos 0=1$이라 적분값은 $2\pi$입니다. $m\ne0$일 때 값이 절반($\pi$)이므로, 이를 보정하기 위해 실수 구면 조화 함수 정의에서 $m\neq0$ 항에만 $\sqrt2$가 붙어 있습니다. ($\sqrt2^2\cdot\pi=2\pi$로 맞춰 줌.)

**(b) $\theta$ 부분** — 버금 르장드르 다항식의 알려진 성질입니다 (증명은 대학 과정이므로 결과만 씁니다).

$$
\int_{-1}^{1}\bigl[P_l^m(x)\bigr]^2dx=\frac{2}{2l+1}\cdot\frac{(l+m)!}{(l-m)!}
$$

여기서 $x=\cos\theta$로 치환하면 $dx=-\sin\theta\,d\theta$가 되어 $dS$의 $\sin\theta$가 자연스럽게 흡수됩니다.

**(c) 합치기.** 전체 크기의 제곱은

$$
\|y_l^m\|^2=(K_l^m)^2\cdot\underbrace{2\pi}_{\varphi}\cdot\underbrace{\frac{2}{2l+1}\frac{(l+|m|)!}{(l-|m|)!}}_{\theta}
$$

이것이 1이 되려면

$$
(K_l^m)^2=\frac{2l+1}{4\pi}\cdot\frac{(l-|m|)!}{(l+|m|)!}
$$

제곱근을 취하면 정확히 카드의 공식이 나옵니다. 즉 $K_l^m$은 "$\theta$ 적분과 $\varphi$ 적분이 만드는 크기의 역수"입니다.

## 5. 손으로 확인해 볼 수 있는 가장 쉬운 경우

$l=0,\;m=0$: $P_0^0=1$, 팩토리얼은 $0!=1$.

$$
K_0^0=\sqrt{\frac{1}{4\pi}}\;\Rightarrow\; y_0^0=\frac{1}{\sqrt{4\pi}}
$$

크기 확인: $\displaystyle\int_{\text{구면}}\Bigl(\frac{1}{\sqrt{4\pi}}\Bigr)^2dS=\frac{1}{4\pi}\cdot 4\pi=1$. 상수 함수를 구 전체 넓이 $4\pi$로 나눠 정규화한 것과 같습니다.

$l=1,\;m=0$: $P_1^0(x)=x$, $\int_{-1}^1x^2dx=\tfrac23$.

$$
K_1^0=\sqrt{\frac{3}{4\pi}},\qquad \|y_1^0\|^2=\frac{3}{4\pi}\cdot 2\pi\cdot\frac23=1\;\checkmark
$$

## 6. 팩토리얼 비율의 직관

$\frac{(l-|m|)!}{(l+|m|)!}$는 $|m|$이 클수록 급격히 작아집니다. 이는 $P_l^m$이 $|m|$이 커질수록 $(1-x^2)^{|m|/2}$와 큰 계수들 때문에 값이 매우 커지는 것을 상쇄하기 위한 것입니다. 원문 코드의 `K(l, m)` 함수가 정확히 이 비율을 계산해 `sqrt`를 취합니다. 절댓값 $|m|$을 쓰는 이유는 $m<0$인 항이 $P_l^{-m}=P_l^{|m|}$을 사용하기 때문입니다.

## 7. 한 줄 정리

$K_l^m$은 구면 넓이($4\pi$), 르장드르 적분($\frac{2}{2l+1}\frac{(l+|m|)!}{(l-|m|)!}$)에서 나오는 크기를 모두 역수로 곱해, 각 기저 함수가 **"길이 1인 서로 수직한 벡터"** 역할을 하게 만드는 상수입니다.
