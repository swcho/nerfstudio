# %% [markdown]
# # 가우시안 분할(split): 자식의 위치와 스케일
#
# gsplat `strategy/ops.py::split` 의 핵심 (splatfacto `_grow_gs` 에서 `scale ≥ densify_size_thresh` 인 것에 적용):
#
# $$\mu' = \mu + R(q)\,(s \odot z),\quad z \sim \mathcal N(0, I),\qquad s' = s / 1.6$$
#
# - 원본을 **삭제**하고 자식 **2개**를 만든다 (`torch.randn(2, N, 3)`).
# - $z$ 를 원본 축 스케일 $s$ 로 늘리고 $R(q)$ 로 회전 → 자식 중심은 **원본 가우시안 분포 $\mathcal N(\mu, \Sigma)$ 에서 샘플링** 한 것과 같다.
# - 스케일은 로그 공간 파라미터라 `torch.log(scales / 1.6)` 로 저장 → $\log s' = \log s - \log 1.6$.
# - `revised_opacity=True` 이면 $\alpha' = 1 - \sqrt{1-\alpha}$ (기본은 원본 불투명도 복사).

# %%
import numpy as np
import plotly.graph_objects as go
import os

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 쿼터니언 → 회전행렬 (gsplat `normalized_quat_to_rotmat`, w,x,y,z 순)

# %%
def quat_to_rotmat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

# z축 기준 30도 회전 (2D 단면을 보기 위해 z축 회전만 사용)
theta = np.deg2rad(30)
q = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
R = quat_to_rotmat(q)
print("R @ R.T ≈ I:", np.allclose(R @ R.T, np.eye(3)), " det:", np.linalg.det(R).round(6))
print("R[:2,:2] =\n", R[:2, :2].round(4))
# 출력: R @ R.T ≈ I: True  det: 1.0
# 출력: R[:2,:2] = [[0.866 -0.5], [0.5 0.866]]  (30도 회전)

# %% [markdown]
# ## 2. 한 가우시안 분할: $\mu' = \mu + R(s \odot z)$, $s' = s/1.6$

# %%
mu = np.array([1.0, 0.5, 0.0])
s = np.array([1.0, 0.4, 0.2])          # 축별 표준편차 (gsplat 의 exp(scales))
log_s = np.log(s)                       # 실제 파라미터는 로그 공간

z = rng.standard_normal((2, 3))         # torch.randn(2, N=1, 3) 에 대응
samples = (R @ (s * z).T).T             # einsum("ij,j,bj->bi") 와 동일
mu_children = mu + samples
s_child = s / 1.6

print("z =\n", z.round(3))
print("mu' =\n", mu_children.round(3))
print("s  =", s, "\ns' =", s_child.round(4))
# 출력: z = [[0.126 -0.132 0.64], [0.105 -0.536 0.362]]
# 출력: mu' = [[1.135 0.517 0.128], [1.198 0.367 0.072]]
# 출력: s = [1. 0.4 0.2]  s' = [0.625 0.25 0.125]

# %% [markdown]
# ## 3. 시각화 (xy 단면): 원본 타원 vs 자식 타원 2개
# $1\sigma$ 타원을 그린다. 자식 중심이 원본 분포 안에 놓이고, 자식 타원이 더 작음(1/1.6)을 확인.

# %%
def ellipse_xy(center, scale, R, k=1.0, n=100):
    t = np.linspace(0, 2*np.pi, n)
    pts = np.stack([scale[0]*np.cos(t), scale[1]*np.sin(t), np.zeros(n)], axis=1) * k
    pts = (R @ pts.T).T + center
    return pts[:, 0], pts[:, 1]

fig = go.Figure()
for k, dash in [(1, "solid"), (2, "dot")]:
    x, y = ellipse_xy(mu, s, R, k)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="gray", dash=dash),
                             name=f"원본 {k}σ"))
fig.add_trace(go.Scatter(x=[mu[0]], y=[mu[1]], mode="markers",
                         marker=dict(color="gray", size=10, symbol="x"), name="μ"))
for i, c in enumerate(["royalblue", "crimson"]):
    x, y = ellipse_xy(mu_children[i], s_child, R, 1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=c), name=f"자식 {i+1} (s/1.6)"))
    fig.add_trace(go.Scatter(x=[mu_children[i, 0]], y=[mu_children[i, 1]], mode="markers",
                             marker=dict(color=c, size=8), name=f"μ'_{i+1}"))
fig.update_layout(title="가우시안 split: 원본 분포에서 샘플한 위치 + s/1.6 스케일 (xy 단면, 1σ)",
                  xaxis_title="x", yaxis_title="y", width=750, height=550,
                  yaxis=dict(scaleanchor="x", scaleratio=1))
_show(fig)
fig.write_image(os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "expy.png"))
# 출력: expy.png 저장

# %% [markdown]
# ## 4. 통계 확인: 자식 중심 $\mu'$ 의 분포 = 원본 공분산 $\Sigma = R\,\mathrm{diag}(s^2)\,R^\top$
# 분할을 여러 번 반복해 $\mu' - \mu$ 의 표본 공분산을 비교한다.

# %%
N = 200_000
Z = rng.standard_normal((N, 3))
D = (R @ (s * Z).T).T                  # μ' - μ
Sigma_true = R @ np.diag(s**2) @ R.T
Sigma_emp = np.cov(D.T)
print("Σ_true =\n", Sigma_true.round(4))
print("Σ_emp  =\n", Sigma_emp.round(4))
print("max |diff| =", np.abs(Sigma_true - Sigma_emp).max().round(4))
print("mean(μ'-μ) =", D.mean(0).round(4))
# 출력: Σ_true = [[0.79 0.3637 0], [0.3637 0.37 0], [0 0 0.04]]
# 출력: Σ_emp  ≈ [[0.7955 0.3666 0], [0.3666 0.3714 0], [0 0 0.04]], max |diff| = 0.0055, mean(μ'-μ) ≈ [0.003 0.002 0]

# %% [markdown]
# ## 5. 로그 공간 확인: $\log(s/1.6) = \log s - \log 1.6$
# gsplat 은 `torch.log(scales / 1.6)` 로 저장한다. 파라미터 관점에서는 단순히 상수 $\log 1.6 \approx 0.47$ 을 빼는 것.

# %%
print("log(s/1.6)        =", np.log(s / 1.6).round(6))
print("log(s) - log(1.6) =", (log_s - np.log(1.6)).round(6))
print("log(1.6) =", np.log(1.6).round(6), " 부피 비율 (1/1.6)^3 =", round((1/1.6)**3, 4))
# 출력: 두 벡터 동일 [-0.470004 -1.386294 -2.079442], log(1.6)=0.470004, 부피 비율 0.2441

# %% [markdown]
# ## 6. (옵션) `revised_opacity=True`: $\alpha' = 1-\sqrt{1-\alpha}$
# 두 자식이 겹쳐 합성될 때 $1-(1-\alpha')^2 = \alpha$ 가 되도록 보정 (arXiv:2404.06109). 기본값은 False로 원본 $\alpha$ 를 그대로 복사.

# %%
alpha = 0.8
alpha_new = 1 - np.sqrt(1 - alpha)
print(f"α={alpha} → α'={alpha_new:.4f},  두 장 합성 1-(1-α')^2 = {1-(1-alpha_new)**2:.4f}")
# 출력: α=0.8 → α'=0.5528, 두 장 합성 = 0.8000
