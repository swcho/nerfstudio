# D3 — 뷰 행렬(viewmat)은 어떻게 구성되는가?

## 한 줄 요약

카메라 포즈 $c2w = [R \mid t]$(camera→world)를 **해석적으로 역변환**해 world→camera 행렬을 만들고, 그 과정에서 nerfstudio(OpenGL 규약)와 gsplat(OpenCV 규약)의 카메라 축 방향 차이를 $y, z$ 축 부호 반전으로 맞춘다.

$$
\text{viewmat} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}
$$

## 실제 구현 (`nerfstudio/models/splatfacto.py`, `get_viewmat`)

```python
def get_viewmat(optimized_camera_to_world):
    R = optimized_camera_to_world[:, :3, :3]  # [B,3,3]
    T = optimized_camera_to_world[:, :3, 3:4]  # [B,3,1]
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, ...)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat
```

입력은 `camera_optimizer.apply_to_camera(camera)`로 보정된 $[B, 3, 4]$ 포즈(D1 단계의 출력), 출력은 $[B, 4, 4]$ 동차 행렬이다. 학습 스텝에서는 $B = 1$.

## 단계별 해설

### 1. 왜 역행렬인가 — c2w vs w2c

nerfstudio의 `Cameras.camera_to_worlds`는 이름 그대로 **카메라 좌표 → 월드 좌표** 변환이다:

$$
\mathbf{x}_w = R\,\mathbf{x}_c + t
$$

여기서 $R$의 열들은 카메라 축($x, y, z$)이 월드에서 어느 방향을 가리키는지, $t$는 카메라 중심의 월드 좌표다.

하지만 래스터화(D5)는 반대 방향이 필요하다. 월드에 놓인 가우시안 중심 `means`를 카메라 앞에 놓고 투영해야 하므로 **월드 → 카메라** 변환(view matrix)을 넘겨야 한다. 위 식을 $\mathbf{x}_c$에 대해 풀면

$$
\mathbf{x}_c = R^{-1}(\mathbf{x}_w - t) = R^\top \mathbf{x}_w - R^\top t
$$

### 2. 왜 `transpose`로 역행렬을 구하는가

$R$은 회전행렬이므로 직교행렬($R^\top R = I$)이고, 따라서 $R^{-1} = R^\top$이다. 일반적인 4×4 `torch.inverse`를 부르는 대신 전치 + 행렬-벡터 곱 한 번으로 끝낼 수 있어 빠르고 수치적으로도 안정적이다(주석의 "analytic matrix inverse"). 결과를 4×4 동차 형태로 채우면

$$
\text{viewmat} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}, \qquad
\text{viewmat} \cdot \begin{bmatrix}\mathbf{x}_w \\ 1\end{bmatrix} = \begin{bmatrix}\mathbf{x}_c \\ 1\end{bmatrix}
$$

### 3. 왜 $y, z$ 축을 뒤집는가 — OpenGL vs OpenCV 규약

| | nerfstudio (OpenGL / Blender 규약) | gsplat (OpenCV / COLMAP 규약) |
|---|---|---|
| $+x$ | 오른쪽 | 오른쪽 |
| $+y$ | **위** | **아래** |
| 카메라가 보는 방향 | **$-z$** | **$+z$** |

두 규약은 $x$축은 같고 $y, z$가 반대이다. `R * [1, -1, -1]`은 $R$의 **열**에 부호를 곱하는 연산(브로드캐스트가 마지막 축, 즉 열 인덱스에 걸린다)이므로

$$
R' = R \begin{bmatrix} 1 & & \\ & -1 & \\ & & -1 \end{bmatrix}
$$

즉 카메라의 $y$축과 $z$축을 각각 반대 방향으로 정의한 새 카메라 프레임을 만든다. 이것은 "OpenCV 카메라 좌표 → 카메라 중심을 공유하는 OpenGL 카메라 좌표"로의 변환 $\mathrm{diag}(1,-1,-1)$을 오른쪽에 합성한 것이며, 결과 $R'$는 여전히 회전행렬($\det = +1$, 두 축을 뒤집었기 때문)이다. 카메라 위치 $t$는 그대로다 — 축 방향만 바뀌고 카메라가 놓인 자리는 같으므로.

이 뒤집기를 **역행렬을 취하기 전에** 수행하는 이유는 간단하다: 역행렬을 취한 뒤라면 행에 부호를 곱해야 하고 $t$쪽도 함께 손봐야 하지만, c2w 단계에서 열을 뒤집으면 이후 공식이 그대로 통한다.

### 4. 함께 넘기는 내부행렬 $K$

D3 단계에서는 `camera.rescale_output_resolution(1/d)`로 카메라를 잠시 $1/d$ 배 줄여 $K$와 $W, H$를 뽑고 원복한다:

$$
K = \begin{bmatrix} f_x/d & 0 & c_x/d \\ 0 & f_y/d & c_y/d \\ 0 & 0 & 1 \end{bmatrix}
$$

`viewmat`은 해상도와 무관하므로 축소 여부에 영향받지 않는다. 최종적으로 gsplat의 `rasterization(..., viewmats=viewmat, Ks=K, width=W, height=H)`에 전달되어, 각 가우시안 중심은 $\mathbf{x}_c = \text{viewmat}\,\mathbf{x}_w$ → $\mathbf{u} = K\,\mathbf{x}_c / z_c$ 로 화면에 투영된다. 이때 $z_c > 0$(OpenCV 규약)이어야 카메라 앞에 있는 점으로 판정되며, `near_plane=0.01`로 컬링된다 — 이것이 축 뒤집기를 빠뜨리면 아무것도 렌더링되지 않는(모든 점이 "카메라 뒤"로 판정되는) 이유다.

## 검증 포인트

- `viewmat[0]`을 출력하면 마지막 행은 항상 `[0, 0, 0, 1]`, 좌상단 3×3은 직교행렬(행끼리 내적 0, 노름 1)이어야 한다.
- `viewmat @ c2w_flipped_homog ≈ I` 를 확인하면 역행렬이 맞는지 볼 수 있다(단, $y,z$ 뒤집힌 c2w 기준).
- 카메라 광학 중심 $t$를 viewmat로 변환하면 $(0,0,0)$이 나와야 한다: $R^\top t - R^\top t = 0$.

## 관련 단계

- **D1** (`camera_optimizer.apply_to_camera`): 여기서 나온 보정된 c2w가 입력.
- **D2/D3**: 다운스케일 $d$에 따라 $K, W, H$ 결정.
- **D5** (`gsplat.rasterization`): `viewmats`, `Ks` 인자로 소비.
