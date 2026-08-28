# %% [markdown]
# # 구면 좌표 → 데카르트 좌표 변환
#
# 단위 구 위의 한 점(=방향 벡터 $\vec n$)은 두 각으로 정해진다.
#
# - $\theta$ (극각, polar angle): $z$축에서 내려온 각, $0 \le \theta \le \pi$
# - $\phi$ (방위각, azimuth): $xy$평면에서 $x$축 기준으로 돈 각, $0 \le \phi < 2\pi$
#
# $$ (x,y,z)=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta) $$
#
# 유도: 높이는 $z=\cos\theta$, $xy$평면에 드리운 그림자 길이는 $\sin\theta$,
# 그 그림자를 $\phi$만큼 돌려 $x,y$로 쪼갠다.
# 이 변환이 있어야 $\cos\theta$, $\sin\theta\cos\phi$ 같은 삼각함수 덩어리가
# 그대로 $z$, $x$가 되어 구면 조화 함수를 **$x,y,z$의 다항식**으로 쓸 수 있다.

# %%
# 필요 패키지: numpy, plotly, kaleido
import os
import numpy as np
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def sph2cart(theta, phi):
    """구면 좌표(단위 반지름) -> 데카르트 좌표."""
    return (np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta))


# 대표적인 방향들을 확인
cases = {
    "theta=0      (북극, +z)": (0.0, 0.0),
    "theta=pi     (남극, -z)": (np.pi, 0.0),
    "theta=pi/2, phi=0    (+x)": (np.pi / 2, 0.0),
    "theta=pi/2, phi=pi/2 (+y)": (np.pi / 2, np.pi / 2),
    "theta=pi/4, phi=pi/4": (np.pi / 4, np.pi / 4),
}
for name, (t, p) in cases.items():
    x, y, z = sph2cart(t, p)
    print(f"{name:28s} -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  |n|={np.hypot(np.hypot(x, y), z):.3f}")
# 출력:
# theta=0      (북극, +z)      -> (+0.000, +0.000, +1.000)  |n|=1.000
# theta=pi     (남극, -z)      -> (+0.000, +0.000, -1.000)  |n|=1.000
# theta=pi/2, phi=0    (+x)    -> (+1.000, +0.000, +0.000)  |n|=1.000
# theta=pi/2, phi=pi/2 (+y)    -> (+0.000, +1.000, +0.000)  |n|=1.000
# theta=pi/4, phi=pi/4         -> (+0.500, +0.500, +0.707)  |n|=1.000

# %% [markdown]
# ## 항등식 확인
# $x^2+y^2+z^2 = \sin^2\theta(\cos^2\phi+\sin^2\phi)+\cos^2\theta = 1$ 이므로
# 언제나 단위 구 위에 놓인다. 역변환은 $\theta=\arccos z,\ \phi=\operatorname{atan2}(y,x)$.

# %%
rng = np.random.default_rng(0)
theta = rng.uniform(0, np.pi, 1000)
phi = rng.uniform(0, 2 * np.pi, 1000)
x, y, z = sph2cart(theta, phi)
print("max |x^2+y^2+z^2 - 1| =", np.abs(x**2 + y**2 + z**2 - 1).max())
theta_back = np.arccos(z)
phi_back = np.mod(np.arctan2(y, x), 2 * np.pi)
print("역변환 오차 theta:", np.abs(theta_back - theta).max(), " phi:", np.abs(phi_back - phi).max())
# 출력:
# max |x^2+y^2+z^2 - 1| = 4.440892098500626e-16
# 역변환 오차 theta: 6.032468361644172e-14  phi: 8.881784197001252e-16

# %% [markdown]
# ## 왜 SH에서 중요한가: $y_1^0 = K\cos\theta = 0.488603\,z$
# 각도 형태의 실수 구면 조화 함수와, 변환으로 얻은 다항식 형태가 같은 값을 준다.
# $l=1$ 세 개는 그저 $y, z, x$에 상수를 곱한 것이고, $l=2$의 $y_2^{-2}=1.092548\,xy$는
# $\sin^2\theta\sin\phi\cos\phi$ 에서 나온다.

# %%
K10 = np.sqrt(3 / (4 * np.pi))          # 0.488603
K2m2 = 0.5 * np.sqrt(15 / np.pi)        # 1.092548
K22 = np.sqrt(5 / (4 * np.pi) / 24)     # K_2^2 = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
# 각도 형태 (원문의 SH(l,m,theta,phi) 정의: m<0 -> sqrt2*K(l,-m)*sin(-m*phi)*P(l,-m,cos theta))
#   P_1^0(cos t) = cos t,  P_2^2(cos t) = 3 sin^2 t
y10_angle = K10 * np.cos(theta)
y2m2_angle = np.sqrt(2) * K22 * np.sin(2 * phi) * 3 * np.sin(theta)**2
# 다항식 형태
y10_poly = 0.488603 * z
y2m2_poly = 1.092548 * x * y
print("K_1^0 =", round(K10, 6), " K_2^-2 =", round(K2m2, 6))
print("y_1^0  각도식 vs 다항식 최대 차:", np.abs(y10_angle - y10_poly).max())
print("y_2^-2 각도식 vs 다항식 최대 차:", np.abs(y2m2_angle - y2m2_poly).max())
# 출력:
# K_1^0 = 0.488603  K_2^-2 = 1.092548
# y_1^0  각도식 vs 다항식 최대 차: 4.880969931564394e-07
# y_2^-2 각도식 vs 다항식 최대 차: 2.1496398139309036e-07  (소수 6자리 반올림 상수 차이)

# %% [markdown]
# ## 시각화
# 단위 구 위에 $\theta$ 등고선(위선)과 $\phi$ 등고선(경선)을 그리고,
# $(\theta,\phi)=(\pi/4,\pi/4)$ 한 점을 벡터로 표시한다.
# 점의 높이가 $\cos\theta$, $xy$평면 그림자 길이가 $\sin\theta$임을 눈으로 확인할 수 있다.

# %%
fig = go.Figure()
# 위선: theta 고정
for t in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):
    p = np.linspace(0, 2 * np.pi, 100)
    xx, yy, zz = sph2cart(t, p)
    zz = np.full_like(xx, zz)  # theta 고정이면 z는 스칼라 -> 배열로 확장
    fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                               line=dict(color="royalblue", width=2), showlegend=False))
# 경선: phi 고정
for p in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    t = np.linspace(0, np.pi, 100)
    xx, yy, zz = sph2cart(t, p)
    fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                               line=dict(color="lightgray", width=1), showlegend=False))
# 예시 점과 벡터 분해
t0, p0 = np.pi / 4, np.pi / 4
px, py, pz = sph2cart(t0, p0)
fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, pz], mode="lines+markers",
                           line=dict(color="red", width=6), marker=dict(size=4), name="n(θ=π/4, φ=π/4)"))
fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, 0], mode="lines",
                           line=dict(color="orange", width=4, dash="dash"), name="xy 그림자 (길이 sinθ)"))
fig.add_trace(go.Scatter3d(x=[px, px], y=[py, py], z=[0, pz], mode="lines",
                           line=dict(color="green", width=4, dash="dash"), name="높이 z = cosθ"))
fig.add_trace(go.Scatter3d(x=[0, 1.3], y=[0, 0], z=[0, 0], mode="lines+text", text=["", "x"],
                           line=dict(color="black", width=3), showlegend=False))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1.3], z=[0, 0], mode="lines+text", text=["", "y"],
                           line=dict(color="black", width=3), showlegend=False))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.3], mode="lines+text", text=["", "z"],
                           line=dict(color="black", width=3), showlegend=False))
fig.update_layout(title="(x,y,z) = (sinθcosφ, sinθsinφ, cosθ)",
                  scene=dict(aspectmode="cube"), width=800, height=700,
                  legend=dict(x=0.02, y=0.98))
_show(fig)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out, scale=2)
print("saved:", out)
# 출력:
# saved: /home/sungwoo/projects/swcho/nerfstudio/fm/.fm/hints/64b81f90-5086-4ab0-a8b1-b69be1ec4918/expy.png
