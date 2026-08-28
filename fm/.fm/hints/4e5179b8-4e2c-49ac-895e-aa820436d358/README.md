# 구면 조화 함수(Spherical Harmonics)의 수학적 정의

## 한 줄 정의

> 구면 조화 함수 $Y_l^m(\theta,\varphi)$는 **구면 좌표계에서 정의되는 함수**로, **구면(단위 구 $S^2$) 위에서 라플라스 방정식 $\nabla^2 f = 0$ 의 해가 이루는 정규 직교 기저**다. 전자기학·양자역학처럼 구면 대칭인 계를 다룰 때 쓰인다.

원문 글(xtozero)이 인용한 위키 정의 그대로다. 이 문장을 세 조각으로 나누어 하나씩 풀어 보자.

## 1. "구면 좌표계에서 정의되는 함수"

3차원 점을 $(x,y,z)$ 대신 반지름 $r$, 천정각(극각) $\theta\in[0,\pi]$, 방위각 $\varphi\in[0,2\pi)$로 표현하는 것이 구면 좌표계다.

$$(x,y,z)=(r\sin\theta\cos\varphi,\; r\sin\theta\sin\varphi,\; r\cos\theta)$$

구면 조화 함수는 $r$에는 의존하지 않고 **방향 $(\theta,\varphi)$만의 함수**다. 즉 단위 구 표면의 각 점에 실수(또는 복소수) 값을 하나씩 대응시키는 함수 $S^2\to\mathbb{C}$다. 방향만의 함수이므로 "구면 위의 함수"라고 부른다. 렌더링에서는 방향 벡터 $\vec n=(x,y,z)$, $|\vec n|=1$ 을 직접 인자로 쓰는 표기 $y_l^m(\vec n)$ 도 자주 쓴다.

## 2. "라플라스 방정식의 해"

라플라스 방정식은 $\nabla^2 f = \frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2}+\frac{\partial^2 f}{\partial z^2}=0$ 이다. 이를 구면 좌표로 바꾸고 해를 $f(r,\theta,\varphi)=R(r)\,Y(\theta,\varphi)$ 꼴로 변수 분리하면, 각도 부분 $Y$ 는 다음 고유값 방정식을 만족해야 한다.

$$\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\,\frac{\partial Y}{\partial\theta}\right)+\frac{1}{\sin^2\theta}\frac{\partial^2 Y}{\partial\varphi^2} = -\,l(l+1)\,Y$$

좌변의 연산자가 **구면 위의 라플라시안**(Laplace–Beltrami 연산자)이고, 구면 전체에서 유한하고 매끄러운 해가 존재하려면 $l=0,1,2,\dots$ 인 정수여야 한다. 이때의 해가 바로 구면 조화 함수 $Y_l^m$이며, 각 $l$에 대해 $m=-l,\dots,l$ 의 $2l+1$개가 독립적으로 존재한다.

다시 $\varphi$ 부분을 분리하면 $e^{im\varphi}$ 가 나오고, 남은 $\theta$ 부분이 **버금 르장드르 다항식** $P_l^m(\cos\theta)$ 가 되어 다음 표준형이 된다.

$$Y_l^m(\theta,\varphi)=\underbrace{\sqrt{\frac{2l+1}{4\pi}\frac{(l-|m|)!}{(l+|m|)!}}}_{K_l^m\;(\text{정규화 상수})}\;P_l^{|m|}(\cos\theta)\;e^{im\varphi}$$

- $l$ : 밴드(band) 인덱스, 차수. 각운동량 양자수에 해당하며 "주파수"의 역할을 한다.
- $m$ : 차수(order), $-l\le m\le l$.
- $P_l^m$ : 버금 르장드르 다항식. 원문은 $P_m^m=(-1)^m(2m-1)!!(1-x^2)^{m/2}$, $P_{m+1}^m=x(2m+1)P_m^m$, $(l-m)P_l^m=x(2l-1)P_{l-1}^m-(l+m-1)P_{l-2}^m$ 세 재귀식으로 이를 계산한다.

$r$ 부분의 해는 $R(r)=r^l$ 또는 $r^{-(l+1)}$ 이므로, 라플라스 방정식의 일반해는 $\sum_{l,m}(a_{lm}r^l+b_{lm}r^{-l-1})Y_l^m$ 으로 전개된다. 이것이 "라플라스 방정식의 해"라는 표현의 뜻이다. 실제로 $r^l Y_l^m$ 은 $x,y,z$의 $l$차 **조화 다항식**(harmonic polynomial: $\nabla^2 p=0$인 다항식)이고, 그것을 단위 구로 제한한 것이 구면 조화 함수라는 동치 정의도 있다. 원문에 나온 실수 SH 표

$$y_0^0=0.282095,\quad y_1^{-1}=0.488603\,y,\quad y_1^0=0.488603\,z,\quad y_1^1=0.488603\,x,$$
$$y_2^{-2}=1.092548\,xy,\quad y_2^{-1}=1.092548\,yz,\quad y_2^0=0.315392(3z^2-1),\quad y_2^1=1.092548\,xz,\quad y_2^2=0.546274(x^2-y^2)$$

가 정확히 이 조화 다항식들이다. 예컨대 $3z^2-1 = 3z^2-(x^2+y^2+z^2)=2z^2-x^2-y^2$ 는 $\nabla^2=-2-2+4=0$ 이다.

## 3. "정규 직교 기저"

구면 위 두 함수의 내적을 구면 전체에 대한 적분으로 정의한다.

$$\langle f,g\rangle=\int_0^{2\pi}\!\!\int_0^{\pi} f\,\overline{g}\,\sin\theta\,d\theta\,d\varphi$$

이 내적에 대해 구면 조화 함수는

$$\langle Y_l^m, Y_{l'}^{m'}\rangle=\delta_{ll'}\delta_{mm'}$$

즉 **서로 다른 것끼리는 직교(내적 0)하고, 자기 자신과의 내적은 1(정규화)** 이다. 앞의 상수 $K_l^m$이 정규화를 맞추는 배율이다. 그리고 **완전성**: 구면 위의 임의의 (제곱적분 가능한) 함수 $f$ 는

$$f(\theta,\varphi)=\sum_{l=0}^{\infty}\sum_{m=-l}^{l} c_{lm}\,Y_l^m(\theta,\varphi),\qquad c_{lm}=\langle f, Y_l^m\rangle$$

로 전개된다. 원문이 "투영(projection)"이라 부른 것이 계수 $c_{lm}$ 을 구하는 이 내적 계산이고, 계수를 곱해 더하는 것이 "복원"이다. 직선 위 주기 함수의 푸리에 급수가 $\{\cos n x,\sin n x\}$ 기저로 전개되듯, **구면 조화 함수는 구면 위 함수의 푸리에 급수**라고 이해하면 정확하다. $l$이 클수록 구면 위에서 더 빠르게 진동하는 고주파 성분이다.

### 실수 구면 조화 함수

$e^{im\varphi}$ 때문에 표준형은 복소수를 반환한다. 렌더링처럼 실수만 필요하면 $\pm m$ 짝을 조합해 실수 기저로 바꾼다(원문 표기).

$$y_l^m(\theta,\varphi)=\begin{cases}\sqrt2\,K_l^m\cos(m\varphi)\,P_l^m(\cos\theta), & m>0\\ \sqrt2\,K_l^m\sin(-m\varphi)\,P_l^{-m}(\cos\theta), & m<0\\ K_l^0\,P_l^0(\cos\theta), & m=0\end{cases}$$

이 역시 정규 직교 기저이며 $\cos m\varphi,\sin m\varphi$가 푸리에 급수의 실수 형태와 대응한다.

## 그림으로 보기

![l=0~4 구면 조화 함수. 초록=양, 빨강=음](fig-1.png)

원문의 그림이다. 각 행이 밴드 $l$, 행 안의 열이 $m=-l,\dots,l$ 이라 $l$행에 $2l+1$개가 있다(1, 3, 5, 7, 9개). 방향 $(\theta,\varphi)$ 마다 $|Y_l^m|$ 만큼 반지름을 늘려 그린 것이라 부호가 바뀌는 곳에서 반지름이 0이 되어 꽃잎(로브) 모양으로 갈라진다. 초록은 $Y>0$, 빨강은 $Y<0$ 영역이다. 옆의 작은 구는 같은 함수를 단위 구 표면 위에 색으로 칠한 것이다.

- $l=0$: 상수 $0.282095=1/\sqrt{4\pi}$ 라서 온통 초록 구 하나 — 부호가 바뀌지 않는다.
- $l=1$: 초록·빨강 두 덩이. $y_1^{-1}\propto y$, $y_1^0\propto z$, $y_1^1\propto x$ 로 각 축 방향에서 부호가 한 번 바뀐다(그림 가운데 것이 위아래로 갈라진 $z$ 축, 좌우가 $y$, $x$ 축).
- $l=2$: 네 잎 꽃잎($xy, yz, xz, x^2-y^2$)과 가운데의 $3z^2-1$(위아래 덩이 + 적도의 빨간 띠).
- $l$이 커질수록 로브 수가 늘어나 "고주파"가 된다는 점이 정규 직교 기저의 주파수 해석과 맞아떨어진다.

## 왜 "구면 대칭인 계"에 쓰이는가

원자의 전자 궤도(양자역학), 점전하 주위의 전위 다중극 전개(전자기학), 지구 중력장 모델 등은 모두 중심으로부터 방향에만 의존하는 성분을 다루는 문제이고, 라플라스 방정식(또는 그 각도 부분)의 해가 자연스럽게 $Y_l^m$ 이 된다. 그래픽스에서는 환경광 $L(\theta,\varphi)$ 같은 "방향의 함수"를 $l\le2$ 의 9개 계수로 압축해 Irradiance Map을 24KB 대신 108Byte(RGB 3 × 9 계수 × 4Byte)로 대체하는 데 쓴다. 3D Gaussian Splatting(splatfacto)에서 각 가우시안의 시점 의존 색을 SH 계수로 저장하는 것도 같은 원리다.

## 요약

| 정의 조각 | 의미 |
|---|---|
| 구면 좌표계에서 정의 | $r$ 무관, 방향 $(\theta,\varphi)$ 만의 함수 $S^2\to\mathbb{C}$ |
| 라플라스 방정식의 해 | $\nabla^2 f=0$ 을 변수 분리했을 때의 각도 부분; $r^lY_l^m$ 은 $l$차 조화 다항식 |
| 정규 직교 기저 | $\langle Y_l^m,Y_{l'}^{m'}\rangle=\delta_{ll'}\delta_{mm'}$, 임의의 구면 함수를 $\sum c_{lm}Y_l^m$ 으로 전개 가능 |
| $l, m$ | 밴드(주파수) $l\ge0$, $-l\le m\le l$, 밴드당 $2l+1$개 |
| 공식 | $Y_l^m=K_l^m P_l^{|m|}(\cos\theta)e^{im\varphi}$, $K_l^m=\sqrt{\frac{2l+1}{4\pi}\frac{(l-|m|)!}{(l+|m|)!}}$ |
