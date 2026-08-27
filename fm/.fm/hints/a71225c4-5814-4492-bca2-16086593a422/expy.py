# %% [markdown]
# # $\Sigma = R S S^\top R^\top$ 를 numpy로 직접 만들어 보기
#
# 3D 가우시안 스플래팅에서 가우시안 $i$의 공분산은 스케일 $s=\exp(\tilde s)$, 쿼터니언 $q$로부터
# $$\Sigma = R(q)\,S\,S^\top R(q)^\top,\qquad S=\mathrm{diag}(s)$$
# 로 만들어집니다. 이 스크립트는 (1) 각 조각을 계산하고, (2) 고유값이 $s^2$임을 확인하고,
# (3) 3D 타원체와 카메라 투영 후 2D 타원을 그립니다.

# %%
# 필요 패키지: numpy, plotly, kaleido (png 저장용)
import numpy as np
import os

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

# %% [markdown]
# ## 1. 스케일: $s = \exp(\tilde s)$
# 학습 파라미터 $\tilde s$는 음수여도 되고, $\exp$를 거치면 항상 양의 반지름이 됩니다.

# %%
s_tilde = np.array([0.0, -0.7, -1.6])          # 학습되는 원시 파라미터 (log 공간)
s = np.exp(s_tilde)                             # 실제 반지름
S = np.diag(s)
print("s_tilde =", s_tilde)
print("s = exp(s_tilde) =", s.round(4))
print("S =\n", S.round(4))
# 출력: s_tilde = [ 0.  -0.7 -1.6]
# 출력: s = exp(s_tilde) = [1.     0.4966 0.2019]
# 출력: S = diag(1, 0.4966, 0.2019)
# 출력 (셀 2): R = [[0.7854 -0.5897 0.1883] [0.5897 0.6202 -0.5173] [0.1883 0.5173 0.8349]]

# %% [markdown]
# ## 2. 회전: 쿼터니언 $q=(w,x,y,z)$ → 회전행렬 $R(q)$
# $$R(q)=\begin{pmatrix}1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)\end{pmatrix}$$
# $q$는 먼저 단위 길이로 정규화합니다 (gsplat도 내부에서 정규화).

# %%
def quat_to_R(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

# z축 기준 40도 회전: q = (cos(θ/2), 0, 0, sin(θ/2)) 를 약간 기울여서(x 성분 추가)
theta = np.deg2rad(40)
q = np.array([np.cos(theta/2), 0.3, 0.0, np.sin(theta/2)])
R = quat_to_R(q)
print("R =\n", R.round(4))
print("R^T R ≈ I ?", np.allclose(R.T @ R, np.eye(3)), " det R =", np.linalg.det(R).round(6))
print("q=(1,0,0,0) → R = I ?", np.allclose(quat_to_R([1, 0, 0, 0]), np.eye(3)))
# 출력: R^T R ≈ I ? True  det R = 1.0
# 출력: q=(1,0,0,0) → R = I ? True

# %% [markdown]
# ## 3. 합성: $\Sigma = R S S^\top R^\top$ 과 고유값 확인
# $M=RS$라 두면 $\Sigma=MM^\top$ 이므로 대칭·양의 준정부호가 자동 보장됩니다.
# 또 $R^\top R=I$ 이므로 $\Sigma R = R\,\mathrm{diag}(s^2)$ → 고유값은 $s^2$, 고유벡터는 $R$의 열.

# %%
Sigma = R @ S @ S.T @ R.T
print("Sigma =\n", Sigma.round(4))
print("대칭?", np.allclose(Sigma, Sigma.T))
eigval, eigvec = np.linalg.eigh(Sigma)
print("고유값 (오름차순) =", eigval.round(6))
print("s^2   (오름차순) =", np.sort(s**2).round(6))
print("고유값 == s^2 ?", np.allclose(np.sort(eigval), np.sort(s**2)))
# 고유벡터가 R의 열과 (부호 무시하고) 일치하는지
cols_R = R[:, np.argsort(s**2)]
print("고유벡터 == R의 열 (부호 무시)?", np.allclose(np.abs(eigvec), np.abs(cols_R)))
# 출력: Sigma = [[0.704 0.369 0.079] [0.369 0.4535 0.1725] [0.079 0.1725 0.1298]]
# 출력: 대칭? True
# 출력: 고유값 (오름차순) = [0.040762 0.246597 1.      ]
# 출력: s^2   (오름차순) = [0.040762 0.246597 1.      ]
# 출력: 고유값 == s^2 ? True
# 출력: 고유벡터 == R의 열 (부호 무시)? True

# %% [markdown]
# ## 4. 왜 $\Sigma$를 직접 학습하면 위험한가
# 대칭행렬의 원소를 조금만 잘못 움직여도 음의 고유값(=음의 분산)이 나옵니다.

# %%
bad = Sigma.copy()
bad[0, 1] += 0.6; bad[1, 0] += 0.6          # 경사하강 한 스텝이 비대각 원소를 크게 밀었다고 가정
print("직접 수정한 Sigma의 고유값 =", np.linalg.eigvalsh(bad).round(4), "→ 음수 존재:", (np.linalg.eigvalsh(bad) < 0).any())
# 반면 s_tilde, q를 아무렇게나 바꿔도 RSS^TR^T는 항상 유효
rng = np.random.default_rng(0)
def rand_sigma():
    Rr, Sr = quat_to_R(rng.normal(size=4)), np.diag(np.exp(rng.normal(size=3)))
    return Rr @ Sr @ Sr.T @ Rr.T
ok = all(np.linalg.eigvalsh(rand_sigma()).min() >= -1e-12 for _ in range(1000))
print("무작위 (s_tilde, q) 1000개 → 모두 고유값 ≥ 0 ?", ok)
# 출력: 직접 수정한 Sigma의 고유값 = [-0.4097  0.1205  1.5766] → 음수 존재: True
# 출력: 무작위 (s_tilde, q) 1000개 → 모두 고유값 ≥ 0 ? True

# %% [markdown]
# ## 5. 카메라 투영: $\Sigma^{2D} = J W \Sigma W^\top J^\top$
# 월드→카메라 변환 $W$(회전), 원근 나눗셈의 야코비안
# $$J=\begin{pmatrix} f_x/z & 0 & -f_x x/z^2\\ 0 & f_y/z & -f_y y/z^2\end{pmatrix}$$
# 를 가우시안 중심 $\mu_c=(x,y,z)$에서 평가합니다.

# %%
mu_world = np.array([0.3, 0.2, 3.0])
W = quat_to_R([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10)), 0, 0])   # 카메라가 x축 기준 20도 기울어짐
fx = fy = 400.0
mu_c = W @ mu_world
x, y, z = mu_c
J = np.array([[fx/z, 0, -fx*x/z**2],
              [0, fy/z, -fy*y/z**2]])
Sigma2D = J @ W @ Sigma @ W.T @ J.T
print("Sigma2D (픽셀^2) =\n", Sigma2D.round(2))
print("2D 고유값 → 화면상 타원 반지름(픽셀) =", np.sqrt(np.linalg.eigvalsh(Sigma2D)).round(2))
# 출력: Sigma2D (픽셀^2) = [[12768.17  6619.41] [ 6619.41  8930.27]]
# 출력: 2D 고유값 → 화면상 타원 반지름(픽셀) = [ 62.91 133.2 ]

# %% [markdown]
# ## 6. 시각화: 3D 타원체($1\sigma$)와 투영된 2D 타원
# 타원체 표면: 단위구의 점 $u$를 $\mu + R S u$ 로 보내면 $\Sigma$의 $1\sigma$ 등고면이 됩니다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

phi, th = np.meshgrid(np.linspace(0, 2*np.pi, 60), np.linspace(0, np.pi, 30))
unit = np.stack([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)], -1)   # (30,60,3)
ell = unit @ (R @ S).T                                                              # 각 점에 RS 적용

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                    subplot_titles=("3D 타원체 Σ = R S Sᵀ Rᵀ (1σ), 축 = R의 열 × s", "투영된 2D 가우시안 Σ²ᴰ = J W Σ Wᵀ Jᵀ"))
fig.add_trace(go.Surface(x=ell[..., 0], y=ell[..., 1], z=ell[..., 2], opacity=0.6, showscale=False, colorscale="Blues"), 1, 1)
for k, c in enumerate(["red", "green", "orange"]):
    a = R[:, k] * s[k]
    fig.add_trace(go.Scatter3d(x=[0, a[0]], y=[0, a[1]], z=[0, a[2]], mode="lines", line=dict(color=c, width=6),
                               name=f"축{k}: s={s[k]:.3f}"), 1, 1)

# 2D: Σ2D 등고선 타원 (1σ, 2σ) + 밀도 히트맵
ev2, evec2 = np.linalg.eigh(Sigma2D)
t = np.linspace(0, 2*np.pi, 200)
for kσ in (1, 2):
    pts = (evec2 * np.sqrt(ev2) * kσ) @ np.stack([np.cos(t), np.sin(t)])
    fig.add_trace(go.Scatter(x=pts[0], y=pts[1], mode="lines", name=f"{kσ}σ 타원", line=dict(width=2)), 1, 2)
g = np.linspace(-300, 300, 121)
gx, gy = np.meshgrid(g, g)
P = np.stack([gx, gy], -1)
inv = np.linalg.inv(Sigma2D)
dens = np.exp(-0.5 * np.einsum("...i,ij,...j->...", P, inv, P))
fig.add_trace(go.Heatmap(x=g, y=g, z=dens, colorscale="Greys", showscale=False, opacity=0.8), 1, 2)
fig.update_xaxes(title="u − μᵤ (픽셀)", row=1, col=2, scaleanchor="y2")
fig.update_yaxes(title="v − μᵥ (픽셀)", row=1, col=2)
fig.update_layout(width=1150, height=520, scene=dict(aspectmode="data"), title="가우시안 공분산: 스케일·회전 → 3D 타원체 → 2D 투영")
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"))
print("saved expy.png")
# 출력: saved expy.png
