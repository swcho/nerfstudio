# %% [markdown]
# # 왜 Irradiance Map은 32x32로 충분한가?
#
# 원본 환경맵(radiance)이 2048x2048처럼 고해상도여도, Diffuse 조명에 쓰이는 **Irradiance**는
#
# $$E(\mathbf n)=\int_{\Omega(\mathbf n)} L_i(\omega)\,(\omega\cdot\mathbf n)\,d\omega$$
#
# 즉 반구 전체에 걸친 **코사인 가중 적분**이다. 적분(=넓은 커널과의 컨볼루션)은 저역 통과 필터이므로
# 결과는 매우 부드럽고, 저해상도로 샘플링한 뒤 **선형 보간**해도 거의 손실이 없다.
#
# 여기서는 개념을 1D 원(circle) 위의 환경으로 축소해 단계적으로 확인한다.
# - 2D 세계: 방향은 각도 $\phi\in[0,2\pi)$ 하나로 표현
# - 환경 radiance $L(\phi)$: 구름·태양처럼 고주파 디테일이 많은 신호
# - Irradiance $E(\mathbf n)=\int \max(0,\cos(\phi-\phi_n))\,L(\phi)\,d\phi$
#
# 필요 패키지: numpy, plotly, kaleido

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 고해상도 "환경맵" 만들기 (2048 샘플)
# 하늘(부드러운 파랑 계조) + 수평선 근처 밝은 띠 + 작은 구름(고주파 잡음) + 아주 작은 태양(스파이크).

# %%
N_HI = 2048
phi = np.linspace(0, 2 * np.pi, N_HI, endpoint=False)

sky = 0.6 + 0.4 * np.clip(np.sin(phi), 0, 1)                     # 위쪽이 밝은 하늘
horizon = 1.5 * np.exp(-((np.sin(phi)) ** 2) / 0.01)               # 수평선 밝은 띠
clouds = 0.25 * np.abs(rng.normal(size=N_HI))                      # 고주파 구름 잡음
sun = 40.0 * np.exp(-((phi - 1.0) ** 2) / (2 * 0.01 ** 2))          # 아주 작은 태양
L = sky + horizon + clouds + sun

print(f"radiance 샘플 수={N_HI}, min={L.min():.3f}, max={L.max():.3f}")
# 출력: radiance 샘플 수=2048, min=0.600, max=41.036

# %% [markdown]
# ## 2. 고해상도 Irradiance를 "정직하게" 계산 (기준값)
# 모든 법선 방향 $\mathbf n$에 대해 클램프 코사인 커널로 적분한다. 이것이 Ground Truth.

# %%
def irradiance(L_src, phi_src, phi_n):
    """phi_n 방향 법선에 대한 irradiance: ∫ max(0,cos(phi-phi_n)) L dphi"""
    dphi = phi_src[1] - phi_src[0]
    w = np.clip(np.cos(phi_src[None, :] - phi_n[:, None]), 0, None)
    return (w * L_src[None, :]).sum(axis=1) * dphi


E_gt = irradiance(L, phi, phi)
print(f"irradiance min={E_gt.min():.3f}, max={E_gt.max():.3f}")
# 출력: irradiance min=1.629, max=3.290

# %% [markdown]
# ## 3. 주파수 관점: radiance vs irradiance 스펙트럼
# 컨볼루션 정리에 의해 $\hat E_k = \hat L_k \cdot \hat A_k$ 이고, 클램프 코사인 커널의 계수 $\hat A_k$는
# $k$가 커질수록 빠르게 0으로 간다 (3D 구면에서의 $\hat A_l$: 3.14, 2.09, 0.79, 0, -0.13, ...).
# 그래서 radiance의 고주파 성분은 irradiance에 거의 남지 않는다.

# %%
def spectrum(x):
    return np.abs(np.fft.rfft(x)) / len(x)


S_L, S_E = spectrum(L), spectrum(E_gt)
# 저주파(k<=8)와 나머지 고주파에 담긴 에너지 비율
def hf_ratio(S, k0=8):
    e = S[1:] ** 2
    return e[k0:].sum() / e.sum()


print(f"radiance   고주파(k>8) 에너지 비율 = {hf_ratio(S_L):.4f}")
print(f"irradiance 고주파(k>8) 에너지 비율 = {hf_ratio(S_E):.6f}")
# 출력: radiance   고주파(k>8) 에너지 비율 = 0.8892
# 출력: irradiance 고주파(k>8) 에너지 비율 = 0.000127

# %% [markdown]
# ## 4. 저해상도로 계산 + 선형 보간 (실제 Irradiance Map 방식)
# 셰이더처럼 **작은 해상도의 법선 방향 몇 개**에 대해서만 irradiance를 계산해 텍셀에 저장하고,
# 런타임에는 `LinearSampler`가 하는 것처럼 선형 보간해 읽는다. 해상도를 바꾸며 오차를 본다.

# %%
def lowres_irradiance_map(res):
    phi_tex = np.linspace(0, 2 * np.pi, res, endpoint=False)
    E_tex = irradiance(L, phi, phi_tex)           # 텍셀 res개에만 저장
    # 원형 선형 보간 (양 끝을 이어 붙임)
    xp = np.concatenate([phi_tex, [2 * np.pi]])
    fp = np.concatenate([E_tex, [E_tex[0]]])
    return np.interp(phi, xp, fp), phi_tex, E_tex


print(f"{'res':>5} | {'max rel err':>11} | {'mean rel err':>12}")
results = {}
for res in [4, 8, 16, 32, 64, 128]:
    E_lo, _, _ = lowres_irradiance_map(res)
    rel = np.abs(E_lo - E_gt) / E_gt
    results[res] = E_lo
    print(f"{res:>5} | {rel.max():>10.2%} | {rel.mean():>11.3%}")
# 출력:   res | max rel err | mean rel err
# 출력:     4 |     18.95% |      5.546%
# 출력:     8 |      7.68% |      1.647%
# 출력:    16 |      4.95% |      0.519%
# 출력:    32 |      0.79% |      0.087%
# 출력:    64 |      0.65% |      0.028%
# 출력:   128 |      0.41% |      0.008%

# %% [markdown]
# 32 샘플(2048의 1/64)만으로 최대 오차 0.8%, 평균 0.09% 수준 — 눈으로 구분할 수 없는 차이다.
# 반대로 **radiance 자체**를 32 샘플로 줄이면 어떻게 될까? 태양·구름이 통째로 사라지거나 뭉개진다.
# 즉 "저해상도가 괜찮은 것"은 어디까지나 *저주파인 irradiance*에 한정된 이야기다.

# %%
idx = np.arange(0, N_HI, N_HI // 32)
L_lo32 = np.interp(phi, np.concatenate([phi[idx], [2 * np.pi]]),
                   np.concatenate([L[idx], [L[idx[0]]]]))
rel_L = np.abs(L_lo32 - L) / L
print(f"radiance를 32샘플로 줄이면 max rel err = {rel_L.max():.1%}, mean = {rel_L.mean():.1%}")
# 출력: radiance를 32샘플로 줄이면 max rel err = 664.3%, mean = 30.3%

# %% [markdown]
# ## 5. 시각화
# 위: 고주파 radiance vs 부드러운 irradiance. 가운데: 누적 에너지 스펙트럼(irradiance는 k≤3에서 거의 100%). 아래: 32 텍셀 + 선형 보간 vs GT.

# %%
E_lo32, phi_tex32, E_tex32 = lowres_irradiance_map(32)
E_lo8, phi_tex8, E_tex8 = lowres_irradiance_map(8)
deg = np.degrees(phi)

fig = make_subplots(
    rows=3, cols=1, vertical_spacing=0.09,
    subplot_titles=(
        "Radiance L(φ) — 고주파 (2048 샘플)  vs  Irradiance E(n) — 저주파",
        "누적 에너지 스펙트럼 : 코사인 적분이 고주파를 제거",
        "Irradiance Map: 32 텍셀 + 선형 보간 vs Ground Truth",
    ),
)
fig.add_trace(go.Scatter(x=deg, y=L, name="Radiance L (clip 5)", line=dict(width=1, color="#888")), 1, 1)
fig.add_trace(go.Scatter(x=deg, y=E_gt, name="Irradiance E (GT)", line=dict(width=3, color="#d62728")), 1, 1)
fig.update_yaxes(range=[0, 5], title_text="값", row=1, col=1)

def cum_energy(S):
    e = S[1:] ** 2
    return np.cumsum(e) / e.sum()


k = np.arange(1, len(S_L))
fig.add_trace(go.Scatter(x=k[:64], y=cum_energy(S_L)[:64], name="Radiance 누적 에너지", line=dict(color="#888")), 2, 1)
fig.add_trace(go.Scatter(x=k[:64], y=cum_energy(S_E)[:64], name="Irradiance 누적 에너지", line=dict(color="#d62728")), 2, 1)
fig.add_vline(x=16, line_dash="dot", line_color="#1f77b4", row=2, col=1,
              annotation_text="32 샘플의 나이퀴스트 한계 k=16", annotation_position="bottom right")
fig.update_yaxes(title_text="누적 에너지 비율", range=[0, 1.05], row=2, col=1)
fig.update_xaxes(title_text="주파수 k (DC 제외)", row=2, col=1)

fig.add_trace(go.Scatter(x=deg, y=E_gt, name="GT (2048)", line=dict(width=4, color="#d62728")), 3, 1)
fig.add_trace(go.Scatter(x=deg, y=E_lo8, name="8 텍셀 + 보간", line=dict(dash="dot", color="#2ca02c")), 3, 1)
fig.add_trace(go.Scatter(x=deg, y=E_lo32, name="32 텍셀 + 보간", line=dict(dash="dash", color="#1f77b4")), 3, 1)
fig.add_trace(go.Scatter(x=np.degrees(phi_tex32), y=E_tex32, mode="markers", name="32 텍셀 값",
                         marker=dict(size=7, color="#1f77b4")), 3, 1)
fig.update_xaxes(title_text="법선 방향 φ (deg)", row=3, col=1)
fig.update_yaxes(title_text="E", row=3, col=1)

fig.update_layout(height=1000, width=900, template="plotly_white",
                  title="Diffuse irradiance는 저주파 → 32x32로 충분한 이유")
_show(fig)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out, scale=2)
print("saved:", out)
# 출력: saved: /home/sungwoo/projects/swcho/nerfstudio/fm/.fm/hints/b303b1ec-f4a0-4ed7-bfbc-26204aff7d73/expy.png

# %% [markdown]
# ## 정리
# - Irradiance = radiance ⊛ 클램프 코사인 로브. 커널 계수 $\hat A_l$이 $l\ge3$부터 급감 → **저주파 신호**.
# - 저주파 신호는 나이퀴스트 관점에서 적은 샘플로 완전히 표현 가능하고, 남는 미세 오차는 **텍스쳐 선형 보간**이 채운다.
# - 그래서 2048x2048 radiance를 입력으로 쓰더라도 결과 Irradiance Map은 32x32(약 24KB)로 충분하며,
#   더 나아가 SH 9계수(108B)로도 거의 같은 결과를 얻는다.
