# %% [markdown]
# # 가우시안 초기 스케일(scales)은 어떻게 정해지는가?
#
# Splatfacto(`populate_modules`)는 SfM 포인트(또는 랜덤 포인트)마다
# **k=3 최근접 이웃까지의 거리 평균**을 구해 초기 스케일로 쓴다:
#
# $$\bar{d}_i = \frac{1}{3}\sum_{j=1}^{3} d_{ij}, \qquad
# s_i = \log \bar{d}_i \;\;(\text{3축에 동일 복제})$$
#
# - 스케일 파라미터는 **log 공간**에 저장되고, 렌더 시 `torch.exp(scales)`로 되돌린다
#   (양수 보장 + 곱셈적 스케일을 덧셈적으로 최적화).
# - 효과: **점이 조밀한 곳 → 이웃 거리 작음 → 작은 가우시안**,
#   **성긴 곳 → 이웃 거리 큼 → 큰 가우시안**. 초기 상태에서 표면을 빈틈없이 덮는다.
#
# 원본 코드 (nerfstudio/models/splatfacto.py):
# ```python
# distances, _ = k_nearest_sklearn(means.data, 3)
# avg_dist = distances.mean(dim=-1, keepdim=True)
# scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
# ```

# %%
# 필요 패키지: torch, scikit-learn, numpy, plotly, kaleido
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

# %% [markdown]
# ## 1. 장난감 포인트 클라우드: 조밀한 영역 + 성긴 영역 (2D)
#
# 왼쪽에는 촘촘한 클러스터(200점, 표준편차 0.15),
# 오른쪽에는 듬성듬성한 클러스터(40점, 표준편차 1.0)를 놓는다.

# %%
rng = np.random.default_rng(42)
dense = rng.normal(loc=[-2.0, 0.0], scale=0.15, size=(200, 2))
sparse = rng.normal(loc=[+2.5, 0.0], scale=1.0, size=(40, 2))
points = np.vstack([dense, sparse]).astype(np.float32)
means = torch.from_numpy(points)  # splatfacto의 means에 해당
print("points:", means.shape, "| dense 200개 + sparse 40개")
# 출력: points: torch.Size([240, 2]) | dense 200개 + sparse 40개

# %% [markdown]
# ## 2. `k_nearest_sklearn(means, 3)` 재현
#
# nerfstudio 구현 그대로: `NearestNeighbors(n_neighbors=k+1)`로 찾은 뒤
# **첫 열(자기 자신, 거리 0)을 버려서** 진짜 이웃 k개의 거리만 남긴다.

# %%
def k_nearest_sklearn(x: torch.Tensor, k: int):
    x_np = x.cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)
    # 자기 자신(distances[:, 0] == 0) 제외
    return (
        torch.tensor(distances[:, 1:], dtype=torch.float32),
        torch.tensor(indices[:, 1:], dtype=torch.int64),
    )


distances, _ = k_nearest_sklearn(means, 3)
print("distances:", distances.shape)          # (N, 3) — 각 점의 3개 이웃 거리
print("첫 점의 3-NN 거리:", distances[0].numpy())
# 출력: distances: torch.Size([240, 3])
# 출력: 첫 점의 3-NN 거리: [0.0155     0.01811043 0.02034284]

# %% [markdown]
# ## 3. 스케일 초기화: 평균 → log → 3축 복제
#
# $$s_i = \log\bar{d}_i,\qquad \text{scales} \in \mathbb{R}^{N\times 3}$$

# %%
avg_dist = distances.mean(dim=-1, keepdim=True)              # (N, 1)
scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))  # (N, 3) — 3축 동일(등방 초기화)
print("scales:", scales.shape)
print("첫 점 scales(log 공간):", scales[0].detach().numpy())
print("exp(scales) = 실제 크기:", torch.exp(scales[0]).detach().numpy())
# 출력: scales: torch.Size([240, 3])
# 출력: 첫 점 scales(log 공간): [-4.0182495 -4.0182495 -4.0182495]
# 출력: exp(scales) = 실제 크기: [0.01798442 0.01798442 0.01798442]

# %% [markdown]
# ## 4. 조밀 vs 성긴 영역 비교
#
# 같은 코드가 밀도에 자동으로 적응하는지 확인한다.

# %%
avg_np = avg_dist.squeeze(-1).numpy()
d_mean, s_mean = avg_np[:200].mean(), avg_np[200:].mean()
print(f"조밀 영역 평균 3-NN 거리: {d_mean:.4f}  -> log = {np.log(d_mean):+.3f}")
print(f"성긴 영역 평균 3-NN 거리: {s_mean:.4f}  -> log = {np.log(s_mean):+.3f}")
print(f"초기 가우시안 크기 비율(성긴/조밀): x{s_mean / d_mean:.1f}")
# 출력: 조밀 영역 평균 3-NN 거리: 0.0340  -> log = -3.382
# 출력: 성긴 영역 평균 3-NN 거리: 0.4800  -> log = -0.734
# 출력: 초기 가우시안 크기 비율(성긴/조밀): x14.1

# %% [markdown]
# ## 5. 시각화
#
# 각 점을 중심으로 반지름 $\exp(s_i)=\bar{d}_i$ 인 원(= 초기 가우시안의 1σ 크기)을 그린다.
# 조밀한 클러스터는 작은 원으로 촘촘히, 성긴 클러스터는 큰 원으로 듬성듬성 덮이는 것을 볼 수 있다.

# %%
import plotly.graph_objects as go

fig = go.Figure()
theta = np.linspace(0, 2 * np.pi, 24)
radii = np.exp(scales[:, 0].detach().numpy())  # = avg_dist

# 초기 가우시안(원) — 하나의 trace로 묶어서 그림
xs, ys = [], []
for (cx, cy), r in zip(points, radii):
    xs.extend((cx + r * np.cos(theta)).tolist() + [None])
    ys.extend((cy + r * np.sin(theta)).tolist() + [None])
fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                         line=dict(color="rgba(99,110,250,0.35)", width=1),
                         name="초기 가우시안 (반지름 = exp(scale))"))
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers",
                         marker=dict(size=3, color="crimson"), name="포인트(means)"))
fig.update_layout(
    title="초기 스케일 = log(3-NN 평균 거리): 조밀→작게, 성긴→크게",
    xaxis_title="x", yaxis_title="y",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    width=900, height=450, template="plotly_white",
)
_show(fig)

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
fig.write_image(os.path.join(_here, "expy.png"), scale=2)
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료

# %% [markdown]
# ## 정리
#
# | 단계 | 코드 | 의미 |
# |---|---|---|
# | 1 | `k_nearest_sklearn(means, 3)` | 각 점의 3-NN 거리 (자기 자신 제외) |
# | 2 | `distances.mean(dim=-1, keepdim=True)` | 이웃 거리 평균 $\bar{d}_i$ → 국소 밀도의 역수 역할 |
# | 3 | `torch.log(avg_dist.repeat(1, 3))` | log 공간 저장 + 3축 동일 복제(등방 초기화) |
#
# 렌더링 시에는 `torch.exp(scales)`로 복원되므로, 학습 시작 시점의 가우시안은
# 정확히 "이웃까지 평균 거리" 크기를 갖는다 — 빈틈도, 과도한 겹침도 적은 출발점.
