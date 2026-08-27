# absgrad에 $(W/2,\ H/2)$를 곱하는 이유 — 단위 환산과 연쇄법칙

## 1. 두 가지 좌표계: 픽셀 좌표와 NDC 좌표

가우시안 하나가 화면에 투영된 중심점을 나타내는 방법이 두 가지 있습니다.

- **픽셀 좌표** $u \in [0, W]$: 화면 왼쪽 끝이 $0$, 오른쪽 끝이 $W$(가로 픽셀 수). 해상도에 따라 범위가 달라집니다.
- **NDC 좌표** $x \in [-1, 1]$: 화면 왼쪽 끝이 $-1$, 오른쪽 끝이 $+1$. 해상도와 무관하게 항상 같은 범위입니다. (NDC = Normalized Device Coordinates, "정규화된 장치 좌표")

둘은 일차함수로 연결됩니다.

$$
u = (x + 1)\cdot \frac{W}{2},
\qquad
v = (y + 1)\cdot \frac{H}{2}
$$

확인: $x=-1 \Rightarrow u=0$, $x=+1 \Rightarrow u=W$. 기울기 $\dfrac{du}{dx} = \dfrac{W}{2}$ 는 "NDC 1단위가 픽셀 몇 개인가"를 뜻합니다. 가로 800픽셀이면 NDC 1단위 = 400픽셀.

gsplat의 래스터라이저는 화면 위치 그래디언트(`means2d`의 `absgrad`)를 **NDC 단위**로 되돌려 줍니다. 즉 $\partial \mathcal L / \partial x$ 입니다.

## 2. 연쇄법칙: 왜 NDC 그래디언트는 해상도에 따라 크기가 달라지나

손실 $\mathcal L$은 픽셀 단위 위치 $u$의 함수이고, $u$는 다시 $x$의 함수입니다. 합성함수의 미분(연쇄법칙)을 쓰면

$$
\frac{\partial \mathcal L}{\partial x}
= \frac{\partial \mathcal L}{\partial u}\cdot\frac{du}{dx}
= \frac{W}{2}\,\frac{\partial \mathcal L}{\partial u}.
$$

이 식을 잘 봐야 합니다. "가우시안을 픽셀 1개만큼 옮겼을 때 손실이 얼마나 변하는가"($\partial\mathcal L/\partial u$)가 **물리적으로 의미 있는 양**입니다. 같은 장면, 같은 가우시안이라면 해상도를 바꿔도 이 값은 거의 비슷하게 유지됩니다(픽셀 하나 어긋난 정도의 오차는 어느 해상도에서든 비슷한 크기의 손실 변화를 만들기 때문).

그런데 우리가 받는 것은 $\partial\mathcal L/\partial x$이고, 여기에는 $W/2$라는 배율이 곱해져 있습니다. 거꾸로 말하면

$$
\frac{\partial \mathcal L}{\partial u} = \frac{2}{W}\,\frac{\partial \mathcal L}{\partial x}
$$

이므로, **같은 픽셀 단위 그래디언트라도 NDC 그래디언트는 $W$가 커질수록 상대적으로 작게, $W$가 작을수록 크게** 나옵니다. 정확히는 NDC 그래디언트 $\partial\mathcal L/\partial x = (W/2)\cdot\partial\mathcal L/\partial u$ 이므로 해상도에 비례합니다. 어느 쪽으로 보든 핵심은 하나입니다: **NDC 그래디언트의 크기는 해상도라는 "우연한" 요인에 끌려다닙니다.**

> 참고: 실제 코드에서는 $\partial\mathcal L/\partial u$ 자체도 해상도에 약하게 의존합니다(손실이 픽셀 평균이므로). 그래도 픽셀 단위로 통일하는 것이 NDC 단위보다 훨씬 안정적입니다. 여기서는 큰 그림(단위 통일)에 집중합니다.

## 3. 해결: $W/2$를 다시 곱해 픽셀 단위로 복원

gsplat `DefaultStrategy._update_state`의 해당 줄은 다음과 같습니다.

```python
# normalize grads to [-1, 1] screen space
grads = info["means2d"].absgrad.clone()
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

주석은 "normalize"라고 썼지만 실제로 하는 일은 **NDC → 픽셀 단위 환산**입니다. 위 연쇄법칙 식에서 $du/dx = W/2$를 곱하는 것이므로, 가우시안이 화면에서 픽셀 단위로 얼마나 움직이길 원하는지에 대응되는 양이 됩니다. 그런 다음 두 성분의 크기(노름)를 누적합니다.

$$
\text{grad2d}_i \mathrel{+}= \Big\|\,\text{absgrad}_i \odot \big(\tfrac W2,\ \tfrac H2\big)\Big\|_2
$$

(`n_cameras`는 한 배치에 여러 카메라를 넣었을 때 평균이 카메라 수로 나뉘어 작아지는 것을 되돌리는 항입니다. splatfacto는 배치 1이므로 1.)

## 4. 이게 왜 중요한가: 해상도 스케줄과 임계값 0.0008

splatfacto는 학습 초반에 **낮은 해상도로 시작해 점점 올리는** 스케줄을 씁니다.

$$
d = 2^{\max(\text{num\_downscales} - \lfloor \text{step}/\text{resolution\_schedule}\rfloor,\ 0)}
$$

step 0에서는 $d=4$, 즉 가로·세로가 원본의 $1/4$입니다. 시간이 지나면 $d=2 \to 1$로 원해상도가 됩니다.

densification은 100스텝마다 "평균 화면공간 그래디언트 $\bar g_i = \text{grad2d}_i / \text{count}_i$ 가 임계값 $\tau_g$ = `densify_grad_thresh` = 0.0008을 넘는가"로 분할/복제할 가우시안을 고릅니다. 이 판정이 학습 전 구간에서 일관돼야 합니다.

- **NDC 단위 그대로 썼다면**: 같은 가우시안, 같은 "픽셀 1개 어긋남"이라도 $d=4$ 시기의 NDC 그래디언트는 $W/8$ 배, $d=1$ 시기에는 $W/2$ 배 — 해상도가 4배 바뀌면 값도 4배 달라집니다. 고정 임계값 0.0008은 초반에는 너무 엄격하고(거의 아무것도 분할 안 됨) 후반에는 너무 느슨해지거나, 그 반대가 됩니다. 스케줄이 바뀔 때마다 임계값을 다시 튠해야 합니다.
- **픽셀 단위로 환산하면**: 각 시기에 실제로 쓰인 `info["width"]`, `info["height"]`(= 그 스텝의 학습 해상도)로 곱하므로, 어느 해상도에서 렌더했든 $\bar g_i$는 "픽셀 몇 개 어긋난 정도의 신호"라는 **같은 물리량**이 됩니다. 따라서 0.0008이라는 숫자 하나가 $d=4$일 때도, $d=1$일 때도 같은 뜻을 가집니다.

## 5. 한 줄 정리

$u = (x+1)\,W/2$ 라는 일차 변환 때문에 연쇄법칙상 $\partial\mathcal L/\partial x = (W/2)\,\partial\mathcal L/\partial u$ 로 NDC 그래디언트에는 해상도 배율이 끼어 있다. $(W/2, H/2)$를 곱해 픽셀 단위 $\partial\mathcal L/\partial u$ 로 되돌려 놓으면, 해상도 스케줄로 $W, H$가 4배 바뀌는 동안에도 `densify_grad_thresh` = 0.0008이 항상 같은 의미(픽셀 단위 위치 오차 신호)를 갖게 된다.
