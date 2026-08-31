# scales와 opacities는 왜 log/logit 공간에 저장될까?

## 1. 문제의 출발점: "값에 조건이 붙어 있는" 파라미터

Splatfacto(3D Gaussian Splatting)의 각 가우시안은 여러 파라미터를 가진다. 그중 두 가지에는 **수학적 제약**이 있다.

- **scale(크기)**: 가우시안의 반지름 같은 것이므로 반드시 **양수**여야 한다. ($s > 0$)
- **opacity(불투명도)**: 얼마나 불투명한지를 나타내는 비율이므로 **0과 1 사이**여야 한다. ($0 < \alpha < 1$)

그런데 학습은 **경사하강법(gradient descent)** 으로 이루어진다. 고등학교에서 배운 미분을 떠올려 보자. 함수 $f$의 최솟값을 찾기 위해 기울기 반대 방향으로 조금씩 이동한다.

$$x_{\text{new}} = x_{\text{old}} - \eta \cdot f'(x_{\text{old}})$$

여기서 $\eta$는 이동 폭(학습률)이다. 문제는 이 갱신 과정이 **$x$가 어떤 값이든 될 수 있다고 가정**한다는 것이다. 예를 들어 scale이 $0.001$인데 기울기 방향으로 $0.002$만큼 빼 버리면 scale이 **음수**가 되어 버린다. 음수 크기의 가우시안은 정의조차 되지 않는다. opacity도 마찬가지로 갱신 한 번에 $1.3$이나 $-0.2$ 같은 무의미한 값이 될 수 있다.

## 2. 해결 아이디어: 제약이 없는 공간에서 최적화하기

조건 "$s > 0$"을 매번 검사하고 잘라내는(clipping) 대신, 더 우아한 방법이 있다.

> **제약이 있는 값 대신, 제약이 없는(unconstrained) 새 변수를 저장하고, 필요할 때 함수를 통과시켜 제약을 만족하는 값으로 변환한다.**

즉, 실수 전체 $(-\infty, \infty)$를 자유롭게 돌아다녀도 되는 변수를 최적화하고, 마지막에 "정답 범위로 보내 주는 함수"를 씌우는 것이다.

### scale: 지수함수와 로그

지수함수 $e^x$를 생각하자. 정의역은 실수 전체, **치역은 항상 양수** $(0, \infty)$이다.

$$s = e^{u}, \qquad u \in (-\infty, \infty) \implies s > 0 \text{ 항상 성립}$$

그래서 실제 크기 $s$ 대신 $u = \ln s$ (log 공간의 값)를 파라미터로 저장한다. $u$가 아무리 크게 음수로 내려가도 $e^u$는 0에 가까워질 뿐 절대 음수가 되지 않는다. 경사하강법이 $u$를 마음껏 갱신해도 안전하다.

### opacity: 시그모이드와 로짓

$(0, 1)$ 구간으로 보내 주는 함수로는 **시그모이드(sigmoid)** 를 쓴다.

$$\sigma(v) = \frac{1}{1 + e^{-v}}$$

$v \to -\infty$이면 $\sigma(v) \to 0$, $v \to +\infty$이면 $\sigma(v) \to 1$이고, 항상 $0 < \sigma(v) < 1$이다. (확률과 통계에서 본 누적분포함수처럼 단조증가하는 S자 곡선이다.)

시그모이드의 **역함수**를 **로짓(logit)** 이라 부른다.

$$\text{logit}(\alpha) = \ln\frac{\alpha}{1-\alpha}$$

실제 불투명도 $\alpha$ 대신 $v = \text{logit}(\alpha)$ (logit 공간의 값)를 저장하면, $v$는 실수 전체를 자유롭게 움직일 수 있다.

### 덤: 미분 관점의 장점

두 함수 모두 매끄럽게 미분 가능하다는 것도 중요하다. 예컨대 $\frac{d}{du}e^u = e^u$이므로, 합성함수 미분(연쇄법칙)으로 기울기가 자연스럽게 흘러 학습이 잘 된다. 경계값( $s=0$이나 $\alpha=1$ ) 근처에서는 갱신이 자동으로 완만해져서, 강제로 잘라내는 방식보다 최적화가 안정적이다.

## 3. 코드에서 확인하기

**저장(초기화) 시 — 역함수를 적용해 unconstrained 공간으로:**

```python
# splatfacto.py — populate_modules()
scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))          # 실제 크기에 log를 취해 저장
opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))  # 초기 불투명도 0.1의 logit을 저장
```

초기 불투명도를 $0.1$로 주고 싶으면, $\text{logit}(0.1) \approx -2.197$을 저장하는 식이다.

**렌더링 시 — 순방향 함수를 적용해 실제 값으로 복원:**

```python
# splatfacto.py — get_outputs()
render, alpha, self.info = rasterization(
    means=means_crop,
    quats=quats_crop,  # rasterization does normalization internally
    scales=torch.exp(scales_crop),                       # log 공간 → 양수 크기
    opacities=torch.sigmoid(opacities_crop).squeeze(-1), # logit 공간 → (0,1) 불투명도
    ...
)
```

정리하면 저장과 사용 사이에 역함수 관계가 성립한다.

$$s = \exp(\underbrace{\ln s}_{\text{저장된 값}}), \qquad \alpha = \sigma(\underbrace{\text{logit}(\alpha)}_{\text{저장된 값}})$$

## 4. 그러면 쿼터니언(quats)은?

회전을 나타내는 쿼터니언에도 "길이가 1이어야 한다"는 제약이 있다. 하지만 코드 주석에 있듯이 (`# rasterization does normalization internally`) 이 정규화는 **rasterization 함수 내부에서 알아서 처리**하므로, 모델 쪽에서 exp나 sigmoid 같은 변환을 따로 해 줄 필요가 없다. 벡터를 자기 길이로 나눠 단위벡터로 만드는 것(정규화)은 언제든 가능하기 때문이다.

## 5. 한 줄 요약

| 파라미터 | 제약 | 저장 공간 | 렌더링 시 변환 |
|---|---|---|---|
| scales | $s > 0$ | log 공간 ($\ln s$) | `torch.exp(scales)` |
| opacities | $0 < \alpha < 1$ | logit 공간 | `torch.sigmoid(opacities)` |
| quats | 단위 길이 | 그대로 저장 | rasterization 내부에서 정규화 |

**경사하강법이 실수 전체를 자유롭게 오갈 수 있도록 제약 없는 공간에 저장하고, 실제로 쓸 때만 exp/sigmoid로 제약을 만족하는 값으로 되돌린다.**
