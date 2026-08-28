# 구면 조화 함수를 "표"로 간단히 구하기

## 질문
버금 르장드르 다항식과 팩토리얼을 직접 계산하지 않고 구면 조화 함수를 간단히 구하는 방법은?

## 핵심 답
**미리 계산해 놓은 구면 조화 함수 표(Table of spherical harmonics)** 를 그대로 가져다 쓴다.
구면 좌표 $(\theta, \phi)$를 데카르트 좌표 $(x, y, z)$로 바꾸면 각 $y_l^m$이 **$x, y, z$의 단순 다항식**이 되므로,
셰이더에서는 상수 곱셈 몇 줄로 끝난다.

---

## 1. 왜 "직접 계산"이 번거로운가

원문(정의)대로 실수 구면 조화 함수를 구하려면 아래 세 요소가 모두 필요하다.

$$ y_{l}^{m}(\theta,\varphi)=\begin{cases} \sqrt{2}K_{l}^{m}\cos(m\varphi)P_{l}^{m}(\cos\theta), & m > 0\\ \sqrt{2}K_{l}^{m}\sin(-m\varphi)P_{l}^{-m}(\cos\theta), & m < 0\\ K_{l}^{0}P_{l}^{0}(\cos\theta), & m = 0 \end{cases} $$

| 요소 | 내용 | 비용 |
|---|---|---|
| $P_l^m(x)$ | 버금 르장드르 다항식. 3개의 재귀식($P_m^m$, $P_{m+1}^m$, 일반 재귀)으로 $P_0^0=1$부터 순차 계산 | 루프 + 이중 계승 $(2m-1)!!$ |
| $K_l^m$ | 정규화 계수 $\sqrt{\frac{2l+1}{4\pi}\frac{(l-\lvert m\rvert)!}{(l+\lvert m\rvert)!}}$ | 팩토리얼 두 번 + sqrt |
| 삼각함수 | $\cos(m\varphi)$, $\sin(-m\varphi)$, $\cos\theta$ | atan2/acos로 각도 복원 후 다시 sin/cos |

이걸 픽셀마다, 9개 기저마다 반복하는 것은 셰이더에서 낭비다. 게다가 방향 벡터 $\vec n$을 가지고 있는데 굳이 $(\theta,\phi)$로 되돌리는 것도 어색하다.

## 2. 해법: 표 + 데카르트 좌표

### (1) 좌표 변환
단위 방향 벡터와 구면 좌표의 관계는

$$ (x,y,z)=(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta) $$

즉 $\cos\theta = z$, $\sin\theta\cos\phi = x$, $\sin\theta\sin\phi = y$ 이므로, 정의식에 등장하는 삼각함수는 전부 $x,y,z$로 치환된다.
(예: $y_1^{0}\propto P_1^0(\cos\theta)=\cos\theta = z$, $y_1^{1}\propto \cos\phi\,P_1^1(\cos\theta)=\cos\phi\sin\theta = x$)

### (2) 상수는 표에서
$K_l^m$과 $\sqrt{2}$, 재귀식에서 나오는 계수를 모두 곱해 **하나의 숫자**로 미리 접어 둔 것이 위키의 [Table of spherical harmonics](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics)다. $l\le 2$ 까지 9개:

$$ \begin{aligned} y_{0}^{0}(\vec n)&=0.282095 \\ y_{1}^{-1}(\vec n)&=0.488603\,y & y_{1}^{0}(\vec n)&=0.488603\,z & y_{1}^{1}(\vec n)&=0.488603\,x \\ y_{2}^{-2}(\vec n)&=1.092548\,xy & y_{2}^{-1}(\vec n)&=1.092548\,yz & y_{2}^{0}(\vec n)&=0.315392\,(3z^2-1) \\ y_{2}^{1}(\vec n)&=1.092548\,xz & y_{2}^{2}(\vec n)&=0.546274\,(x^2-y^2) \end{aligned} $$

상수의 출처를 확인해 보면: $0.282095=\frac{1}{2}\sqrt{1/\pi}$, $0.488603=\frac{1}{2}\sqrt{3/\pi}$, $1.092548=\frac{1}{2}\sqrt{15/\pi}$, $0.315392=\frac{1}{4}\sqrt{5/\pi}$, $0.546274=\frac{1}{4}\sqrt{15/\pi}$ — 모두 $K_l^m$(과 $\sqrt2$)을 계산해 접은 값이다.

### (3) 결과: 셰이더 코드
원문의 HLSL 구현은 표를 그대로 옮긴 것이다. 루프도, 팩토리얼도, 삼각함수도 없다.

```hlsl
void ShFunctionL2( float3 v, out float Y[9] )
{
    Y[0] = 0.282095f;                              // Y_00
    Y[1] = 0.488603f * v.y;                        // Y_1-1
    Y[2] = 0.488603f * v.z;                        // Y_10
    Y[3] = 0.488603f * v.x;                        // Y_11
    Y[4] = 1.092548f * v.x * v.y;                  // Y_2-2
    Y[5] = 1.092548f * v.y * v.z;                  // Y_2-1
    Y[6] = 0.315392f * ( 3.f * v.z * v.z - 1.f );  // Y_20
    Y[7] = 1.092548f * v.x * v.z;                  // Y_21
    Y[8] = 0.546274f * ( v.x * v.x - v.y * v.y );  // Y_22
}
```

입력 `v`는 정규화된 방향(노멀) 벡터 하나면 충분하다.

## 3. 그림으로 보는 "표에 있는 함수들"

![l=0~4 실수 구면 조화 함수의 형태 (초록: 양, 빨강: 음)](fig-1.png)

- 그림의 각 행이 $l$, 행 안의 열이 $m=-l\ldots l$ 이다. 위 표(코드)에 대응하는 것은 **위 세 행($l=0,1,2$)의 9개** 도형이다.
- $l=0$: 초록 구 하나 = 상수 $0.282095$ (방향에 무관).
- $l=1$: 초록/빨강 두 덩이가 각각 y, z, x 축을 따라 갈라진다 = $y$, $z$, $x$에 비례하는 1차 다항식(부호가 축 방향으로 바뀜).
- $l=2$: 네 잎(xy, yz, xz, x²−y²) 또는 가운데 띠가 있는 아령 모양($3z^2-1$) = 2차 다항식. 그림의 잎 개수/방향이 곱해진 좌표 항과 정확히 일치한다.
- $l\ge3$은 표에는 있지만, Irradiance Map 근사에는 $l\le2$의 9개(RGB 3채널 × 9 = 27개 계수)로 충분해 사용하지 않는다.

## 4. 한 줄 정리
> 재귀식·팩토리얼로 $P_l^m$, $K_l^m$을 매번 구하지 말고, 표에 접혀 있는 **상수 × $x,y,z$ 다항식**을 그대로 쓴다. 입력은 방향 벡터, 출력은 9개 float — 셰이더가 매우 간단해진다.

## 참고
- 원문: `spherical-harmonics.md` "구면 조화 함수" 절 및 `ShFunctionL2` 코드
- [Wikipedia — Table of spherical harmonics](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics)
- Robin Green, *Spherical Harmonic Lighting: The Gritty Details* (2003)
