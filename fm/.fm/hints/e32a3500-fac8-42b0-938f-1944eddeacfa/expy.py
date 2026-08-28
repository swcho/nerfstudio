# %% [markdown]
# # `ShFunctionL2(float3 v, out float Y[9])` — 무엇을 계산하는가?
#
# 이 셰이더 함수는 **단위 방향 벡터 $\vec v=(x,y,z)$**에 대해
# 차수 $l \le 2$ 인 **실수 구면 조화 함수(real SH) 9개**의 값을 채운다.
#
# | idx | $y_l^m$ | 식 |
# |---|---|---|
# | 0 | $y_0^0$ | $0.282095$ |
# | 1 | $y_1^{-1}$ | $0.488603\,y$ |
# | 2 | $y_1^{0}$ | $0.488603\,z$ |
# | 3 | $y_1^{1}$ | $0.488603\,x$ |
# | 4 | $y_2^{-2}$ | $1.092548\,xy$ |
# | 5 | $y_2^{-1}$ | $1.092548\,yz$ |
# | 6 | $y_2^{0}$ | $0.315392\,(3z^2-1)$ |
# | 7 | $y_2^{1}$ | $1.092548\,xz$ |
# | 8 | $y_2^{2}$ | $0.546274\,(x^2-y^2)$ |
#
# 상수들은 정규화 계수다:
# $$0.282095=\tfrac{1}{2}\sqrt{\tfrac{1}{\pi}},\quad
# 0.488603=\tfrac{1}{2}\sqrt{\tfrac{3}{\pi}},\quad
# 1.092548=\tfrac{1}{2}\sqrt{\tfrac{15}{\pi}},\quad
# 0.315392=\tfrac{1}{4}\sqrt{\tfrac{5}{\pi}},\quad
# 0.546274=\tfrac{1}{4}\sqrt{\tfrac{15}{\pi}}$$
#
# 왜 9개만? Irradiance는 저주파 신호라 $l\le2$ 로도 오차가 1% 수준이므로,
# RGB × 9 = 27개 계수로 Irradiance Map을 근사할 수 있다.

# %%
# 필요 패키지: numpy, plotly, kaleido
import os
import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def ShFunctionL2(v):
    """HLSL ShFunctionL2 의 numpy 이식. v: (..., 3) 단위 벡터 -> (..., 9)"""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    Y = np.empty(v.shape[:-1] + (9,))
    Y[..., 0] = 0.282095                              # Y_00
    Y[..., 1] = 0.488603 * y                          # Y_1-1
    Y[..., 2] = 0.488603 * z                          # Y_10
    Y[..., 3] = 0.488603 * x                          # Y_11
    Y[..., 4] = 1.092548 * x * y                      # Y_2-2
    Y[..., 5] = 1.092548 * y * z                      # Y_2-1
    Y[..., 6] = 0.315392 * (3.0 * z * z - 1.0)        # Y_20
    Y[..., 7] = 1.092548 * x * z                      # Y_21
    Y[..., 8] = 0.546274 * (x * x - y * y)            # Y_22
    return Y


# %% [markdown]
# ## 1단계: 상수가 정말 정규화 계수인지 확인

# %%
consts = {
    "0.282095": 0.5 * np.sqrt(1 / np.pi),
    "0.488603": 0.5 * np.sqrt(3 / np.pi),
    "1.092548": 0.5 * np.sqrt(15 / np.pi),
    "0.315392": 0.25 * np.sqrt(5 / np.pi),
    "0.546274": 0.25 * np.sqrt(15 / np.pi),
}
for k, val in consts.items():
    print(f"{k} ≈ {val:.6f}")
# 출력:
# 0.282095 ≈ 0.282095
# 0.488603 ≈ 0.488603
# 1.092548 ≈ 1.092548
# 0.315392 ≈ 0.315392
# 0.546274 ≈ 0.546274

# %% [markdown]
# ## 2단계: 몇 개 방향에 대해 직접 호출해 보기
# +z 방향에서는 $y_1^0$, $y_2^0$ 만 살아있고, x/y 성분이 0이므로 나머지 홀수항은 0이다.

# %%
np.set_printoptions(precision=4, suppress=True)
for name, v in [("+z", [0, 0, 1]), ("+x", [1, 0, 0]),
                ("(1,1,1)/√3", np.array([1, 1, 1]) / np.sqrt(3))]:
    print(f"v={name:>10}: Y =", ShFunctionL2(np.asarray(v, float)))
# 출력:
# v=        +z: Y = [0.2821 0.     0.4886 0.     0.     0.     0.6308 0.     0.    ]
# v=        +x: Y = [0.2821 0.     0.     0.4886 0.     0.     -0.3154 0.     0.5463]
# v=(1,1,1)/√3: Y = [0.2821 0.2821 0.2821 0.2821 0.3642 0.3642 0.     0.3642 0.    ]

# %% [markdown]
# ## 3단계: 정규직교성 검증
# 구면 위에서 $\int y_i\,y_j\,d\omega=\delta_{ij}$ 이어야 한다.
# 원문의 컴퓨트 셰이더처럼 $(\theta,\phi)$ 격자에 $\sin\theta$ 가중치를 곱해 수치 적분한다.

# %%
n1, n2 = 256, 128
phi = (np.arange(n1) + 0.5) * 2 * np.pi / n1
theta = (np.arange(n2) + 0.5) * np.pi / n2
TH, PH = np.meshgrid(theta, phi, indexing="ij")
dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
Y = ShFunctionL2(dirs).reshape(-1, 9)
w = (np.sin(TH) * (2 * np.pi / n1) * (np.pi / n2)).reshape(-1)
gram = (Y * w[:, None]).T @ Y
print("Gram 행렬 (≈ 단위행렬):")
print(gram)
print("최대 오차 |G - I| =", np.abs(gram - np.eye(9)).max())
# 출력:
# Gram 행렬 (≈ 단위행렬):
# [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.]
#  ... (대각 ≈1, 비대각 ≈0, 최대 1e-4 수준 이산화 오차)
# 최대 오차 |G - I| = 0.000128

# %% [markdown]
# ## 4단계: 원문 컴퓨트 셰이더처럼 $L_{lm}$ 투영 → 재구성
# 간단한 "하늘은 밝고 땅은 어두운" 환경광 $L(\vec n)=\max(0,n_z)$ 을 9개 계수로 투영하고
# 다시 $\sum_i L_i\,y_i(\vec n)$ 으로 복원해 얼마나 잘 맞는지 본다.

# %%
L = np.clip(dirs[..., 2], 0, None).reshape(-1)           # radiance
coeffs = (Y * (L * w)[:, None]).sum(0)                    # L_lm = Σ L y sinθ dθdφ
recon = Y @ coeffs
print("계수 L_lm:", coeffs)
print("복원 RMS 오차:", np.sqrt(np.mean((recon - L) ** 2)))
# 출력:
# 계수 L_lm: [ 0.8863 -0.      1.0234 -0.     -0.     -0.      0.4955 -0.     -0.    ]
# 복원 RMS 오차: 0.0388

# %% [markdown]
# ## 5단계: 시각화 — 9개 기저함수를 구면 위에 그리기
# 각 소구면의 색이 $y_l^m(\vec n)$ 값(빨강 +, 파랑 −)이다. 행: $l=0,1,2$.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = ["Y_0^0", "Y_1^-1", "Y_1^0", "Y_1^1", "Y_2^-2", "Y_2^-1", "Y_2^0", "Y_2^1", "Y_2^2"]
positions = {0: (1, 2), 1: (2, 1), 2: (2, 2), 3: (2, 3),
             4: (3, 1), 5: (3, 2), 6: (3, 3), 7: (3, 4), 8: (3, 5)}
fig = make_subplots(rows=3, cols=5, specs=[[{"type": "surface"}] * 5] * 3,
                    subplot_titles=[labels[i] if (r, c) in positions.values() else ""
                                    for r in range(1, 4) for c in range(1, 6)
                                    for i in [next((k for k, p in positions.items() if p == (r, c)), 0)]])
Ysph = ShFunctionL2(dirs)
for i in range(9):
    r, c = positions[i]
    fig.add_trace(go.Surface(x=dirs[..., 0], y=dirs[..., 1], z=dirs[..., 2],
                             surfacecolor=Ysph[..., i], colorscale="RdBu_r",
                             cmin=-0.6, cmax=0.6, showscale=(i == 8)), row=r, col=c)
fig.update_layout(title="ShFunctionL2: l≤2 실수 구면 조화 함수 9개 (구면 위 값)",
                  height=800, width=1300, margin=dict(l=0, r=0, t=60, b=0))
fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                  aspectmode="cube")
_show(fig)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "expy.png")
fig.write_image(out, scale=1)
print("saved", out)
# 출력: saved .../expy.png
