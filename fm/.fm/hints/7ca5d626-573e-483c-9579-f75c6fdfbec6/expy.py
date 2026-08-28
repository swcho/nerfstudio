# %% [markdown]
# # 반구 적분 → 구면 좌표 적분: 수치로 확인하기
#
# 목표 식:
#
# $$\frac{1}{\pi}\int_{\Omega} L(\omega)(\omega\cdot\mathbf{n})\,d\omega
# \;=\;\frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L(\theta,\phi)\cos\theta\,\sin\theta\,d\theta\,d\phi$$
#
# 단계:
# 1. 구면 좌표 $\to$ 방향 벡터, $\omega\cdot\mathbf{n}=\cos\theta$ 확인
# 2. $d\omega=\sin\theta\,d\theta\,d\phi$ — 반구 넓이 $2\pi$로 검산
# 3. 균일 조명에서 결과가 $L$이 되는지($\frac{1}{\pi}$ 정규화) 확인
# 4. 상한 $\pi/2$(반구) vs $\pi$(전체 구) 비교
# 5. 리만합(원문 셰이더 방식) vs 몬테카를로(균일 방향 샘플) 교차 검증
# 6. 시각화: 반구 위 피적분 함수 $\cos\theta\sin\theta$

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


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. 구면 좌표 → 방향 벡터, 내적은 $\cos\theta$
#
# $$\omega=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta),\qquad \mathbf{n}=(0,0,1)$$

# %%
def direction(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)

n = np.array([0.0, 0.0, 1.0])
for th, ph in [(0.0, 0.0), (np.pi / 4, 1.0), (np.pi / 2, 2.0), (np.pi / 3, 5.0)]:
    w = direction(th, ph)
    print(f"theta={th:.3f} phi={ph:.1f} |w|={np.linalg.norm(w):.4f} "
          f"w·n={w @ n:.4f} cos(theta)={np.cos(th):.4f}")
# 출력:
# theta=0.000 phi=0.0 |w|=1.0000 w·n=1.0000 cos(theta)=1.0000
# theta=0.785 phi=1.0 |w|=1.0000 w·n=0.7071 cos(theta)=0.7071
# theta=1.571 phi=2.0 |w|=1.0000 w·n=0.0000 cos(theta)=0.0000
# theta=1.047 phi=5.0 |w|=1.0000 w·n=0.5000 cos(theta)=0.5000

# %% [markdown]
# ## 2. $d\omega=\sin\theta\,d\theta\,d\phi$ 검산
#
# 반구 표면적은 $2\pi$. $\sin\theta$를 빼면 $2\pi\cdot\frac{\pi}{2}=\pi^2$로 틀린다.

# %%
def riemann(f, theta_max, n1=400, n2=400):
    """원문 셰이더와 같은 이중 리만합. f(theta, phi) -> 값.
    d_phi = 2pi/n1, d_theta = theta_max/n2 (칸 중심에서 샘플)."""
    phi = (np.arange(n1) + 0.5) * (2 * np.pi / n1)
    theta = (np.arange(n2) + 0.5) * (theta_max / n2)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return f(TH, PH).sum() * (2 * np.pi / n1) * (theta_max / n2)

area_with_sin = riemann(lambda th, ph: np.sin(th), np.pi / 2)
area_without = riemann(lambda th, ph: np.ones_like(th), np.pi / 2)
print(f"sin 포함  : {area_with_sin:.5f}  (정답 2*pi = {2*np.pi:.5f})")
print(f"sin 미포함: {area_without:.5f}  (pi^2 = {np.pi**2:.5f}, 틀림)")
# 출력:
# sin 포함  : 6.28319  (정답 2*pi = 6.28319)
# sin 미포함: 9.86960  (pi^2 = 9.86960, 틀림)

# %% [markdown]
# ## 3. 균일 조명 $L\equiv 1$에서 결과는 1이어야 한다 ($\frac{1}{\pi}$ 정규화)
#
# $$\frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,d\phi
# =\frac{1}{\pi}\cdot 2\pi\cdot\tfrac12 = 1$$

# %%
def irradiance(L, theta_max=np.pi / 2, n1=400, n2=400):
    return riemann(lambda th, ph: L(th, ph) * np.cos(th) * np.sin(th), theta_max, n1, n2) / np.pi

E_uniform = irradiance(lambda th, ph: np.ones_like(th))
print(f"균일 조명 L=1 -> E = {E_uniform:.6f}")
# 출력:
# 균일 조명 L=1 -> E = 1.000003

# %% [markdown]
# ## 4. 상한 $\pi/2$ (반구) vs $\pi$ (전체 구)
#
# 전체 구에서 $\cos\theta$를 그대로 적분하면 아래 반구가 음수로 기여해 상쇄된다.
# 원문 후반부처럼 $\max(\cos\theta,0)$을 쓰면 전체 구 적분이 반구 적분과 같아진다.

# %%
E_half = irradiance(lambda th, ph: np.ones_like(th), theta_max=np.pi / 2)
E_full_raw = riemann(lambda th, ph: np.cos(th) * np.sin(th), np.pi) / np.pi
E_full_clamped = riemann(lambda th, ph: np.maximum(np.cos(th), 0) * np.sin(th), np.pi) / np.pi
print(f"반구  (theta<=pi/2)             : {E_half:.6f}")
print(f"전체구 cos 그대로 (theta<=pi)    : {E_full_raw:.6f}  <- 상쇄되어 0")
print(f"전체구 max(cos,0) (theta<=pi)    : {E_full_clamped:.6f}  <- 반구와 동일")
# 출력:
# 반구  (theta<=pi/2)             : 1.000003
# 전체구 cos 그대로 (theta<=pi)    : 0.000000  <- 상쇄되어 0
# 전체구 max(cos,0) (theta<=pi)    : 1.000010  <- 반구와 동일

# %% [markdown]
# ## 5. 비균일 조명: 리만합 vs 몬테카를로 교차 검증
#
# 조명 $L(\omega)=1+2\,\max(\omega\cdot\mathbf{s},0)$ (방향 $\mathbf{s}$ 쪽이 밝은 하늘).
# 몬테카를로는 **구면 좌표를 쓰지 않고** 3D에서 균일하게 방향을 뽑아
# $\frac{1}{\pi}\int_\Omega L(\omega)(\omega\cdot\mathbf{n})d\omega \approx \frac{1}{\pi}\cdot\frac{2\pi}{N}\sum L(\omega_k)(\omega_k\cdot\mathbf{n})$ 로 계산한다.
# 둘이 일치하면 $\sin\theta$ 야코비안이 옳다는 독립적 증거다.

# %%
s = np.array([0.6, 0.0, 0.8])  # 밝은 쪽 방향 (단위 벡터)

def L_sky_dir(w):                       # w: (..., 3)
    return 1.0 + 2.0 * np.maximum(w @ s, 0.0)

def L_sky(th, ph):                      # 구면 좌표 버전
    return L_sky_dir(direction(th, ph))

E_riemann = irradiance(L_sky, n1=800, n2=800)

rng = np.random.default_rng(0)
N = 2_000_000
v = rng.normal(size=(N, 3))
v /= np.linalg.norm(v, axis=1, keepdims=True)
v[:, 2] = np.abs(v[:, 2])               # 위쪽 반구로 접기 (반구에서 균일 분포)
E_mc = (2 * np.pi / N) * np.sum(L_sky_dir(v) * v[:, 2]) / np.pi

print(f"리만합(구면좌표)   : {E_riemann:.5f}")
print(f"몬테카를로(3D 균일): {E_mc:.5f}")
print(f"상대 오차          : {abs(E_riemann - E_mc) / E_riemann * 100:.3f} %")
# 출력:
# 리만합(구면좌표)   : 2.10283
# 몬테카를로(3D 균일): 2.10339
# 상대 오차          : 0.027 %

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: 반구 위에 피적분 함수 $\cos\theta\sin\theta$ (조명 $L=1$일 때 각 방향의 기여도)를 색으로.
# 천정($\theta=0$)은 $\cos$이 커도 $\sin$이 0이라 기여 0, 지평선($\theta=\pi/2$)은 $\cos$이 0이라 기여 0,
# 중간 $\theta=\pi/4$에서 최대.
#
# 오른쪽: $\theta$에 따른 $\cos\theta$, $\sin\theta$, $\cos\theta\sin\theta$ 곡선과 상한 $\pi/2$.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

th = np.linspace(0, np.pi / 2, 60)
ph = np.linspace(0, 2 * np.pi, 120)
TH, PH = np.meshgrid(th, ph, indexing="ij")
XYZ = direction(TH, PH)
W = np.cos(TH) * np.sin(TH)

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "xy"}]],
                    subplot_titles=("반구 위 피적분 함수 cosθ·sinθ", "θ에 따른 인수들 (상한 π/2)"))
fig.add_trace(go.Surface(x=XYZ[..., 0], y=XYZ[..., 1], z=XYZ[..., 2], surfacecolor=W,
                         colorscale="Viridis", cmin=0, cmax=0.5,
                         colorbar=dict(title="cosθ sinθ", x=0.45, len=0.8)), row=1, col=1)
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.3], mode="lines+text", text=["", "n"],
                           line=dict(color="red", width=6), showlegend=False), row=1, col=1)

t = np.linspace(0, np.pi, 300)
fig.add_trace(go.Scatter(x=t, y=np.cos(t), name="cosθ (Lambert)"), row=1, col=2)
fig.add_trace(go.Scatter(x=t, y=np.sin(t), name="sinθ (면적 보정)"), row=1, col=2)
fig.add_trace(go.Scatter(x=t, y=np.cos(t) * np.sin(t), name="cosθ·sinθ", line=dict(width=4)), row=1, col=2)
fig.add_shape(type="line", x0=np.pi / 2, x1=np.pi / 2, y0=-1, y1=1, xref="x", yref="y",
              line=dict(dash="dash", color="gray"))
fig.add_annotation(x=np.pi / 2, y=1.0, xref="x", yref="y", text="θ=π/2 (반구 경계)",
                   showarrow=False, xanchor="left")
fig.update_xaxes(title_text="θ (rad)", tickvals=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                 ticktext=["0", "π/4", "π/2", "3π/4", "π"], row=1, col=2)
fig.update_layout(width=1200, height=550, scene=dict(aspectmode="data"),
                  legend=dict(x=0.75, y=0.95))

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=1)
print("saved:", png_path)
_show(fig)
# 출력:
# saved: /home/sungwoo/projects/swcho/nerfstudio/fm/.fm/hints/7ca5d626-573e-483c-9579-f75c6fdfbec6/expy.png
