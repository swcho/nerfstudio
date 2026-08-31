# %% [markdown]
# # `get_viewmat` 탐구: c2w → world-to-camera(viewmat) 변환
#
# splatfacto.py의 `get_viewmat`은 camera-to-world(c2w) 행렬을
# gsplat이 요구하는 world-to-camera(viewmat) 행렬로 바꾼다.
#
# 1. 무작위지만 **유효한** c2w를 만든다 (QR 분해로 정규직교 회전 $R$ + 평행이동 $T$)
# 2. 소스 코드 그대로 `get_viewmat`을 구현한다
# 3. $\text{viewmat} = \begin{pmatrix} R^T & -R^T T \\ \mathbf{0}^T & 1 \end{pmatrix}$ 이
#    실제로 (축 뒤집은) c2w의 역행렬과 같은지 `torch.linalg.inv` / `numpy.linalg.inv`로 검증한다
# 4. 축 뒤집기 전/후의 카메라 축을 3D plotly 장면으로 시각화한다

# %%
# 필요 패키지: torch, numpy, plotly, kaleido
import numpy as np
import torch


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)

# %% [markdown]
# ## 1. 무작위 유효 c2w 만들기
#
# 아무 행렬이나 회전 행렬이 되는 것은 아니다. 회전 행렬은 열들이 서로 수직인
# 단위벡터(정규직교)여야 한다. 무작위 행렬을 **QR 분해**하면 Q가 정규직교 행렬이
# 되므로 이를 회전으로 쓴다 ($\det Q = -1$이면 한 열을 뒤집어 $+1$로 맞춘다).

# %%
def random_c2w(n: int) -> torch.Tensor:
    """(n, 4, 4) 무작위 camera-to-world 행렬 배치를 만든다."""
    A = torch.randn(n, 3, 3, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)              # Q: 정규직교
    det = torch.linalg.det(Q)              # det = ±1
    Q[:, :, 0] *= det.sign().unsqueeze(-1)  # det = +1 (진짜 회전)으로 보정
    T = torch.randn(n, 3, 1, dtype=torch.float64) * 2.0

    c2w = torch.zeros(n, 4, 4, dtype=torch.float64)
    c2w[:, :3, :3] = Q
    c2w[:, :3, 3:4] = T
    c2w[:, 3, 3] = 1.0
    return c2w


c2w = random_c2w(4)
R = c2w[:, :3, :3]
print("R^T R ≈ I ?", torch.allclose(R.transpose(1, 2) @ R,
                                    torch.eye(3, dtype=torch.float64).expand(4, 3, 3),
                                    atol=1e-12))
print("det(R) =", torch.linalg.det(R).numpy().round(12))
# 출력: R^T R ≈ I ? True
# 출력: det(R) = [1. 1. 1. 1.]

# %% [markdown]
# ## 2. `get_viewmat` — 소스 코드 그대로 (65–81행, `@torch_compile()`만 제외)
#
# - `R * [[1, -1, -1]]` : $R$의 y·z **열** 부호 반전 → OpenGL 관례(카메라가 $-z$를 봄, $+y$ 위)를
#   gsplat/OpenCV 관례(카메라가 $+z$를 봄, $-y$ 위)로 변경
# - `R.transpose(1, 2)` : 정규직교이므로 $R^{-1} = R^T$
# - `-torch.bmm(R_inv, T)` : $-R^T T$
#
# $$p_w = R p_c + T \;\Longrightarrow\; p_c = R^T p_w - R^T T$$

# %%
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


viewmat = get_viewmat(c2w)
print(viewmat[0].numpy().round(4))
# 출력: [[-0.8909  0.3854 -0.2403 -0.0819]
# 출력:  [ 0.1527  0.7525  0.6407 -4.7087]
# 출력:  [ 0.4278  0.5341 -0.7292 -3.6293]
# 출력:  [ 0.      0.      0.      1.    ]]

# %% [markdown]
# ## 3. 수치 역행렬과 대조 검증
#
# `get_viewmat(c2w)`는 **축을 뒤집은** c2w의 역행렬이어야 한다.
# 뒤집은 c2w를 직접 만들어 `torch.linalg.inv`, `numpy.linalg.inv`와 비교한다.

# %%
# 축 뒤집은 c2w (열 y, z 부호 반전 — 평행이동 T는 그대로)
c2w_flipped = c2w.clone()
c2w_flipped[:, :3, 1:3] *= -1

inv_torch = torch.linalg.inv(c2w_flipped)
inv_numpy = np.linalg.inv(c2w_flipped.numpy())

print("torch.linalg.inv 와 일치?", torch.allclose(viewmat, inv_torch, atol=1e-12))
print("numpy.linalg.inv 와 일치?", np.allclose(viewmat.numpy(), inv_numpy, atol=1e-12))
# viewmat @ c2w_flipped = I 도 확인
prod = viewmat @ c2w_flipped
print("viewmat @ c2w_flipped ≈ I ?",
      torch.allclose(prod, torch.eye(4, dtype=torch.float64).expand(4, 4, 4), atol=1e-12))
# 출력: torch.linalg.inv 와 일치? True
# 출력: numpy.linalg.inv 와 일치? True
# 출력: viewmat @ c2w_flipped ≈ I ? True

# %% [markdown]
# 반대로 축을 **뒤집지 않은** 원본 c2w의 역행렬과는 다르다는 것도 확인해 두자
# (y·z 뒤집기가 실제로 무언가를 바꾼다는 증거).

# %%
print("원본 c2w의 역행렬과 같은가?",
      torch.allclose(viewmat, torch.linalg.inv(c2w), atol=1e-9))
# 출력: 원본 c2w의 역행렬과 같은가? False

# %% [markdown]
# ## 4. 시각화: 축 뒤집기 전/후의 카메라 축
#
# c2w의 회전 열들은 "월드에서 본 카메라 축"이다.
# - 왼쪽: OpenGL/nerfstudio 관례 — 파란 $-z$ 쪽을 보고, 초록 $+y$가 위
# - 오른쪽: y·z 열을 뒤집은 gsplat/OpenCV 관례 — 파란 $+z$ 쪽을 보고, $y$는 아래
#
# 같은 물리적 카메라인데 축의 이름표(부호)만 바뀐 것임을 화살표 방향으로 확인한다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cam = c2w[0]                      # 카메라 하나만 사용
origin = cam[:3, 3].numpy()
axes_gl = cam[:3, :3].numpy()     # OpenGL 관례 축 (열벡터)
axes_cv = axes_gl * np.array([1, -1, -1])  # y·z 열 뒤집기

colors = ["#d62728", "#2ca02c", "#1f77b4"]   # x=빨강, y=초록, z=파랑
names = ["x", "y", "z"]

fig = make_subplots(
    rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=["OpenGL/nerfstudio (c2w 원본)", "gsplat/OpenCV (y·z 뒤집음)"],
)

def add_axes(fig, axes, col, suffix):
    for i in range(3):
        tip = origin + axes[:, i]
        fig.add_trace(go.Scatter3d(
            x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
            mode="lines+text", line=dict(color=colors[i], width=8),
            text=["", f"{names[i]}{suffix}"], textposition="top center",
            textfont=dict(color=colors[i], size=14), showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[axes[0, i]], v=[axes[1, i]], w=[axes[2, i]],
            sizemode="absolute", sizeref=0.15, anchor="tip",
            colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False,
        ), row=1, col=col)
    # 카메라 위치
    fig.add_trace(go.Scatter3d(
        x=[origin[0]], y=[origin[1]], z=[origin[2]], mode="markers",
        marker=dict(size=5, color="black"), showlegend=False,
    ), row=1, col=col)

add_axes(fig, axes_gl, 1, "")
add_axes(fig, axes_cv, 2, "'")

scene = dict(aspectmode="cube",
             xaxis=dict(range=[origin[0] - 1.5, origin[0] + 1.5]),
             yaxis=dict(range=[origin[1] - 1.5, origin[1] + 1.5]),
             zaxis=dict(range=[origin[2] - 1.5, origin[2] + 1.5]))
fig.update_layout(scene=scene, scene2=scene, width=1000, height=520,
                  title="카메라 축: x는 그대로, y·z만 반전 (같은 카메라, 다른 관례)",
                  margin=dict(l=10, r=10, t=80, b=10))

_show(fig)

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
fig.write_image(os.path.join(_here, "expy.png"), scale=2)  # kaleido 필요
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료

# %% [markdown]
# ## 정리
#
# - `get_viewmat`은 c2w의 y·z 열을 뒤집어 gsplat 관례로 맞춘 뒤,
#   정규직교성($R^{-1}=R^T$)을 이용해 역행렬을 $\left[\,R^T \mid -R^T T\,\right]$로
#   **해석적으로** 계산한다.
# - 결과는 `torch.linalg.inv` / `numpy.linalg.inv`로 구한 수치 역행렬과 완전히 일치하지만,
#   전치 + 행렬곱 한 번이라 더 싸고 안정적이며, 실제 코드에서는 `@torch_compile()`로 한층 더 가속된다.
