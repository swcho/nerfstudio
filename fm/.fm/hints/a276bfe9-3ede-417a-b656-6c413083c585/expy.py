# %% [markdown]
# # 리만합으로 바꾼 Irradiance 계산식 검증
#
# 목표식:
# $$\frac{1}{\pi}\int_{0}^{2\pi}\!\!\int_{0}^{\pi/2} L(\theta,\phi)\cos\theta\sin\theta\,d\theta\,d\phi
# \;\approx\;\frac{1}{\pi}\frac{2\pi}{n_1}\frac{\pi/2}{n_2}\sum\sum L\cos\theta\sin\theta
# \;=\;\frac{\pi}{n_1 n_2}\sum_{\phi}^{n_1}\sum_{\theta}^{n_2} L\cos\theta\sin\theta$$
#
# 단계별로 (1) 상수 정리, (2) 균일 하늘($L\equiv1$)에서 수렴, (3) 비균일 하늘에서 셰이더식 vs 정밀적분, (4) 시각화.

# %%
# 필요 패키지: numpy, plotly, kaleido
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

OUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")

# %% [markdown]
# ## 1. 상수 정리: $\frac{1}{\pi}\cdot\frac{2\pi}{n_1}\cdot\frac{\pi/2}{n_2} = \frac{\pi}{n_1 n_2}$ 인지 수치로 확인

# %%
for n1, n2 in [(10, 5), (64, 32), (251, 63)]:
    lhs = (1 / np.pi) * (2 * np.pi / n1) * ((np.pi / 2) / n2)
    rhs = np.pi / (n1 * n2)
    print(f"n1={n1:4d} n2={n2:3d}  lhs={lhs:.10f}  rhs={rhs:.10f}  same={np.isclose(lhs, rhs)}")
# 출력: n1=  10 n2=  5  lhs=0.0628318531  rhs=0.0628318531  same=True
# 출력: n1=  64 n2= 32  lhs=0.0015339808  rhs=0.0015339808  same=True
# 출력: n1= 251 n2= 63  lhs=0.0001986715  rhs=0.0001986715  same=True

# %% [markdown]
# ## 2. 셰이더식 구현
# 원문 HLSL과 같은 구조: 두 반복문으로 $L\cos\theta\sin\theta$를 누적하고, 마지막에 `PI * sum / numSample`.

# %%
def shader_irradiance(L, n1, n2):
    """L(theta, phi) -> 밝기. 리만합(왼쪽 끝점) 방식으로 셰이더식을 계산."""
    phis = np.arange(n1) * (2 * np.pi / n1)
    thetas = np.arange(n2) * ((np.pi / 2) / n2)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    acc = np.sum(L(TH, PH) * np.cos(TH) * np.sin(TH))
    num_sample = n1 * n2
    return np.pi * acc / num_sample

# 균일 하늘 L≡1: 해석적 정답은 (1/π)·2π·(1/2) = 1
L_uniform = lambda th, ph: np.ones_like(th)
for n in [4, 16, 64, 256]:
    print(f"n1=n2={n:4d}  shader={shader_irradiance(L_uniform, n, n):.6f}  exact=1.000000")
# 출력: n1=n2=   4  shader=0.948059  exact=1.000000
# 출력: n1=n2=  16  shader=0.996785  exact=1.000000
# 출력: n1=n2=  64  shader=0.999799  exact=1.000000
# 출력: n1=n2= 256  shader=0.999987  exact=1.000000

# %% [markdown]
# 칸 수를 키우면 1로 수렴한다. 오차는 $\theta$ 방향의 리만합 오차이며, $\phi$ 방향은 피적분함수가
# $\phi$에 무관하므로 오차가 없다. 가중치 $\cos\theta\sin\theta$가 양 끝($0,\pi/2$)에서 모두 0이라
# 왼쪽 끝점 합이 사다리꼴 규칙과 같아져 오차가 $O(1/n_2^2)$로 빠르게 줄어든다.

# %% [markdown]
# ## 3. 비균일 하늘: 위쪽이 밝은 하늘 $L = 1 + \cos\theta + 0.5\sin\theta\cos\phi$
# 정밀한 참값은 $\theta,\phi$ 각각 5000칸의 중점 규칙으로 계산해 비교.

# %%
L_sky = lambda th, ph: 1.0 + np.cos(th) + 0.5 * np.sin(th) * np.cos(ph)

def midpoint_reference(L, n=5000):
    dphi, dth = 2 * np.pi / n, (np.pi / 2) / n
    phis = (np.arange(n) + 0.5) * dphi
    thetas = (np.arange(n) + 0.5) * dth
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    return (1 / np.pi) * np.sum(L(TH, PH) * np.cos(TH) * np.sin(TH)) * dth * dphi

ref = midpoint_reference(L_sky)
print(f"reference = {ref:.6f}   (해석값: 1 + 2/3 = {1 + 2/3:.6f})")
ns = [4, 8, 16, 32, 64, 128, 256]
errs = []
for n in ns:
    v = shader_irradiance(L_sky, n, n)
    errs.append(abs(v - ref))
    print(f"n1=n2={n:4d}  shader={v:.6f}  abs_err={errs[-1]:.2e}")
# 출력: reference = 1.666667   (해석값: 1 + 2/3 = 1.666667)
# 출력: n1=n2=   4  shader=1.588546  abs_err=7.81e-02
# 출력: n1=n2=   8  shader=1.647328  abs_err=1.93e-02
# 출력: n1=n2=  16  shader=1.661844  abs_err=4.82e-03
# 출력: n1=n2=  32  shader=1.665462  abs_err=1.21e-03
# 출력: n1=n2=  64  shader=1.666365  abs_err=3.01e-04
# 출력: n1=n2= 128  shader=1.666591  abs_err=7.53e-05
# 출력: n1=n2= 256  shader=1.666648  abs_err=1.88e-05

# %% [markdown]
# 오차가 $n$이 2배가 될 때마다 약 1/4로 줄어 $O(1/n^2)$ 수렴을 보인다. 원문 셰이더의
# `SampleDelta = 0.025`는 $n_1\approx251,\ n_2\approx63$에 해당한다.

# %% [markdown]
# ## 4. 시각화
# 왼쪽: 피적분함수의 가중치 $\cos\theta\sin\theta$ 와 리만합 막대(왼쪽 끝점, $n_2=8$).
# 오른쪽: 균일/비균일 하늘에서 칸 수에 따른 수렴.

# %%
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("가중치 cosθ·sinθ 와 리만합 (n₂=8)",
                                    "칸 수 n에 따른 셰이더식 수렴"))
th_fine = np.linspace(0, np.pi / 2, 400)
fig.add_trace(go.Scatter(x=th_fine, y=np.cos(th_fine) * np.sin(th_fine), mode="lines",
                         name="cosθ sinθ", line=dict(color="#1f77b4", width=3)), row=1, col=1)
n2 = 8
dth = (np.pi / 2) / n2
th_left = np.arange(n2) * dth
fig.add_trace(go.Bar(x=th_left + dth / 2, y=np.cos(th_left) * np.sin(th_left), width=dth,
                     name="리만합 막대(왼쪽 끝점)", marker=dict(color="rgba(255,127,14,0.45)",
                     line=dict(color="#ff7f0e", width=1))), row=1, col=1)

vals_uniform = [shader_irradiance(L_uniform, n, n) for n in ns]
vals_sky = [shader_irradiance(L_sky, n, n) for n in ns]
fig.add_trace(go.Scatter(x=ns, y=vals_uniform, mode="lines+markers", name="균일 하늘 (정답 1)",
                         line=dict(color="#2ca02c")), row=1, col=2)
fig.add_trace(go.Scatter(x=ns, y=vals_sky, mode="lines+markers", name="비균일 하늘 (정답 5/3)",
                         line=dict(color="#d62728")), row=1, col=2)
fig.add_hline(y=1.0, line=dict(dash="dash", color="#2ca02c"), row=1, col=2)
fig.add_hline(y=ref, line=dict(dash="dash", color="#d62728"), row=1, col=2)
fig.update_xaxes(title_text="θ (rad)", row=1, col=1)
fig.update_yaxes(title_text="가중치", row=1, col=1)
fig.update_xaxes(title_text="n1 = n2", type="log", row=1, col=2)
fig.update_yaxes(title_text="π/(n1 n2) · Σ L cosθ sinθ", row=1, col=2)
fig.update_layout(width=1100, height=450, template="plotly_white",
                  title_text="리만합 → π/(n₁n₂) Σ L cosθ sinθ : 적분의 이산화와 수렴")
_show(fig)
fig.write_image(str(OUT_DIR / "expy.png"), scale=2)
print("saved:", OUT_DIR / "expy.png")
# 출력: saved: /home/sungwoo/projects/swcho/nerfstudio/fm/.fm/hints/a276bfe9-3ede-417a-b656-6c413083c585/expy.png
