# `get_viewmat`은 무엇을 하는 함수인가 — 고교 수학에서 출발하는 설명

splatfacto(가우시안 스플래팅)의 `get_viewmat`(splatfacto.py 65–81행)는 한 줄로 요약하면:

> **camera-to-world(c2w) 행렬을 받아, gsplat 라이브러리가 요구하는 world-to-camera(viewmat) 행렬로 바꿔 주는 함수.** 그 과정에서 (1) 좌표축 관례 차이 때문에 y·z축을 뒤집고, (2) 역행렬을 일반 공식이 아니라 "해석적(analytic)"으로 빠르게 계산하며, (3) `@torch_compile()`로 컴파일해 속도를 더 올린다.

이걸 제대로 이해하려면 고등학교 기하 벡터에서 시작해 네 단계만 쌓으면 된다.

---

## 1단계: 벡터와 좌표계 — "같은 점, 다른 주소"

공간의 한 점 $P$는 절대적인 존재지만, 그 **좌표**는 "어느 좌표계에서 읽었는가"에 따라 달라진다.

- **월드 좌표계**: 장면(scene) 전체의 기준. 원점과 $x, y, z$축이 고정되어 있다.
- **카메라 좌표계**: 카메라 렌즈 중심이 원점이고, 카메라가 보는 방향·위쪽 방향이 축이 되는 좌표계. 카메라가 움직이면 이 좌표계도 통째로 움직인다.

렌더링을 하려면 "월드에 놓인 가우시안들이 **카메라 눈에는** 어디에 보이는가"를 알아야 하므로, **월드 좌표 → 카메라 좌표** 변환(world-to-camera)이 필요하다. 그런데 nerfstudio가 들고 있는 것은 반대 방향인 **camera-to-world**(카메라 좌표 → 월드 좌표) 행렬이다. 그래서 **역변환**을 구해야 한다 — 이것이 `get_viewmat`의 존재 이유다.

## 2단계: 회전을 행렬로 쓰기

고교 과정에서 평면 위의 점 $(x, y)$를 원점 중심으로 $\theta$만큼 회전하면

$$
x' = x\cos\theta - y\sin\theta, \qquad y' = x\sin\theta + y\cos\theta
$$

이다. 이 두 식을 "숫자표에 벡터를 곱하는" 형태로 묶어 쓴 것이 **회전 행렬**이다:

$$
\begin{pmatrix} x' \\ y' \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}}_{R}
\begin{pmatrix} x \\ y \end{pmatrix}
$$

행렬 곱의 규칙은 "결과의 각 성분 = 행렬의 해당 **행**과 벡터의 내적". 3차원에서도 똑같이 $3\times 3$ 행렬 $R$로 회전을 표현한다.

**핵심 관찰**: $R$의 **열(column)들은 "원래의 축들이 회전 후 어디로 가는가"** 를 나타낸다. 예컨대 첫 번째 열은 $R \begin{pmatrix}1\\0\\0\end{pmatrix}$, 즉 $x$축 단위벡터가 이동한 곳이다. c2w 행렬에서 $R$의 세 열은 각각 **카메라의 x·y·z축을 월드 좌표로 표현한 벡터**다. 이 사실이 4단계의 "축 뒤집기"를 이해하는 열쇠가 된다.

## 3단계: 회전 행렬의 마법 — $R^{-1} = R^T$

회전은 길이와 각도를 보존하므로, $R$의 세 열은 서로 수직인 단위벡터들이다(**정규직교**, orthonormal). 이때 전치행렬 $R^T$(행과 열을 뒤바꾼 것)를 곱해 보면:

$$
(R^T R)_{ij} = (\text{$R$의 $i$번째 열}) \cdot (\text{$R$의 $j$번째 열}) =
\begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}
$$

내적이 이렇게 나오는 이유가 바로 "단위벡터(자기 자신과의 내적 1) + 서로 수직(내적 0)"이다. 즉 $R^T R = I$, 다시 말해

$$
\boxed{R^{-1} = R^T}
$$

일반적인 $n \times n$ 역행렬은 가우스 소거 등으로 힘들게 구해야 하지만, **회전 행렬은 그냥 뒤집어(전치) 놓기만 하면 역행렬이다.** 이것이 코드의 `R_inv = R.transpose(1, 2)` 한 줄이다.

## 4단계: 이동까지 합치기 — $4\times 4$ 동차(homogeneous) 변환

카메라 자세는 회전 $R$만으로는 부족하고 위치(평행이동) $T$도 필요하다. 점 $p$의 카메라 좌표 $p_c$와 월드 좌표 $p_w$ 사이의 관계는

$$
p_w = R\, p_c + T
$$

("카메라 좌표를 회전시키고 카메라 위치만큼 옮기면 월드 좌표"). 이걸 행렬 **하나**로 쓰기 위해 벡터 끝에 1을 붙인 4차원 벡터를 쓴다:

$$
\begin{pmatrix} p_w \\ 1 \end{pmatrix}
=
\underbrace{\begin{pmatrix} R & T \\ \mathbf{0}^T & 1 \end{pmatrix}}_{\text{c2w}}
\begin{pmatrix} p_c \\ 1 \end{pmatrix}
$$

이렇게 하면 "회전 + 이동"이 행렬 곱 한 번이 되고, 변환의 합성은 행렬 곱, **역변환은 역행렬**이 된다.

### 역변환을 손으로 풀기

$p_w = R p_c + T$를 $p_c$에 대해 풀면:

$$
p_c = R^{-1}(p_w - T) = R^T p_w - R^T T
$$

즉 world-to-camera 행렬은

$$
\text{viewmat} = \begin{pmatrix} R^T & -R^T T \\ \mathbf{0}^T & 1 \end{pmatrix}
$$

이것이 코드 그대로다:

```python
R_inv = R.transpose(1, 2)        # R^T
T_inv = -torch.bmm(R_inv, T)     # -R^T T
viewmat[:, :3, :3] = R_inv
viewmat[:, :3, 3:4] = T_inv
viewmat[:, 3, 3] = 1.0           # 마지막 행 (0 0 0 1)
```

`torch.linalg.inv`로 $4\times4$ 역행렬을 일반적으로 구할 수도 있지만, 구조를 아는 덕분에 **전치 한 번 + 행렬-벡터 곱 한 번**으로 끝난다. 더 빠르고 수치적으로도 안정적이다(일반 역행렬 알고리즘의 반올림 오차가 없다).

## 그런데 왜 y·z축을 뒤집는가?

코드에서 역행렬 계산 **전에** 이런 줄이 있다:

```python
R = R * torch.tensor([[[1, -1, -1]]], ...)   # R의 2·3번째 열 부호 반전
```

이것은 $R$의 **y열과 z열의 부호를 뒤집는** 연산이다. 2단계에서 봤듯 $R$의 열 = 카메라 축이므로, 이는 **카메라의 y축과 z축 방향을 반대로 정의**하는 것과 같다.

이유는 그래픽스 진영마다 카메라 좌표계 관례가 다르기 때문이다:

| 관례 | 카메라가 보는 방향 | 위쪽 |
|---|---|---|
| **OpenGL / nerfstudio** | $-z$ (z축의 반대쪽을 봄) | $+y$ |
| **OpenCV / COLMAP / gsplat** | $+z$ (z축 방향을 봄) | $-y$ (y가 아래) |

두 관례는 y축과 z축이 정확히 반대다. nerfstudio의 c2w는 OpenGL식인데 gsplat은 OpenCV식 viewmat을 기대하므로, 역행렬을 만들기 전에 카메라의 y·z축을 뒤집어 "gsplat이 생각하는 카메라"로 바꿔 놓는 것이다. x축은 두 관례에서 같으므로 그대로 둔다(그래서 곱하는 값이 $[1, -1, -1]$).

참고로 열 두 개의 부호를 뒤집은 행렬도 여전히 정규직교(열들이 단위벡터 + 서로 수직)이므로, 3단계의 $R^{-1}=R^T$ 트릭은 뒤집은 뒤에도 그대로 성립한다.

## 마지막 조각: `@torch_compile()`

이 함수는 렌더링할 때마다(학습 스텝마다) 호출된다. 데코레이터 `@torch_compile()`은 PyTorch가 이 작은 텐서 연산 나열을 한 번 분석해 융합된(최적화된) 커널로 컴파일해 두게 한다. 수학적 의미는 바꾸지 않고 **속도만** 올리는 장치다.

## 요약

$$
\text{viewmat} \;=\; \Big(\text{c2w를 gsplat 관례로 y·z축 뒤집기}\Big)^{-1}
\;=\; \begin{pmatrix} R^T & -R^T T \\ \mathbf{0}^T & 1 \end{pmatrix}
$$

- **입력**: nerfstudio(OpenGL 관례)의 camera-to-world 행렬 배치 `(N, 4, 4)`
- **출력**: gsplat(OpenCV 관례)의 world-to-camera 행렬 배치 `(N, 4, 4)`
- **기법**: 정규직교성 덕분에 역행렬을 $R^T$, $-R^T T$로 해석적으로 계산 + `torch_compile`로 가속
