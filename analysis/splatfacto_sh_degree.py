# %% [markdown]
# # SH 차수 이해하기 — splatfacto의 `features_dc` / `features_rest`
#
# [splatfacto_train_step.py](splatfacto_train_step.py) G단계에서 `features_rest`의 grad가 0으로 나오고,
# "step 0은 `sh_degree_to_use=0` 이라 고차 밴드가 forward에 안 쓰임"이라는 주석만 달려 있었습니다.
# 이 스크립트는 그 한 줄을 펼칩니다 — **SH 차수가 무엇이고, 왜 나눠 저장하고, 왜 천천히 푸는가.**
#
# ```
# ① SH란 무엇인가 → ② basis를 눈으로 → ③ C0와 RGB2SH → ④ 차수를 올리면 뭐가 좋아지나
#                 → ⑤ splatfacto의 SH 파이프라인 → ⑥ sh_degree_interval → ⑦ 학습된 모델의 실제 SH
# ```
#
# **실행 환경**: conda env `nerfstudio`. ①~④는 CPU만으로 돌고, ⑤는 gsplat CUDA 커널이 필요합니다.
# ⑦은 30k 스텝 완주한 체크포인트가 있으면 실행되고 없으면 자동으로 건너뜁니다.
#
# 참조 코드: [nerfstudio/utils/spherical_harmonics.py](../nerfstudio/utils/spherical_harmonics.py),
# [nerfstudio/models/splatfacto.py](../nerfstudio/models/splatfacto.py)

# %%
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _find_repo() -> Path:
    """스크립트/Interactive Window/노트북 어디서 실행해도 저장소 루트를 찾는다 (pyproject.toml 기준)."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists() and (d / "nerfstudio").is_dir():
            return d
    raise RuntimeError(f"nerfstudio 저장소 루트를 못 찾음 (시작점 {start})")


REPO = _find_repo()
os.chdir(REPO)
torch.manual_seed(0)

# matplotlib 한글 폰트 — 설치된 CJK 폰트 중 첫 번째를 사용 (없으면 경고만 뜨고 □로 표시됨)
import matplotlib.font_manager as fm

_ko = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("Noto Sans CJK", "Noto Sans KR", "Nanum", "Malgun"))]
if _ko:
    plt.rcParams["font.family"] = sorted(set(_ko))[0]
plt.rcParams["axes.unicode_minus"] = False

from nerfstudio.utils.spherical_harmonics import (
    MAX_SH_DEGREE,
    RGB2SH,
    SH2RGB,
    components_from_spherical_harmonics,
    num_sh_bases,
)

print("repo:", REPO, "| MAX_SH_DEGREE =", MAX_SH_DEGREE, "| CUDA:", torch.cuda.is_available())


# %% [markdown]
# ## ① SH란 무엇인가 — 구면 위의 푸리에 급수
#
# 가우시안 하나의 색이 **보는 방향에 따라 달라져야** 합니다. 금속 표면의 하이라이트, 나뭇잎의 반투과, 물의 반사 — 전부 시점 의존입니다.
# 그런데 "방향 → 색" 함수는 정의역이 **구면** $S^2$ 입니다. 이 함수를 어떻게 몇 개의 숫자로 저장할까요?
#
# 1차원 주기함수를 $\sin/\cos$ 급수로 펼치듯, 구면 위 함수는 **구면조화함수(spherical harmonics)** 로 펼칩니다:
#
# $$
# f(\mathbf d) \;\approx\; \sum_{l=0}^{L}\ \sum_{m=-l}^{l} c_{lm}\, Y_{lm}(\mathbf d)
# $$
#
# - $\mathbf d$ : 단위 방향 벡터 (카메라 → 가우시안)
# - $Y_{lm}$ : **고정된** basis 함수. 학습 대상이 아니라 $\mathbf d$ 만 넣으면 값이 나오는 상수 함수
# - $c_{lm}$ : **학습되는 계수**. splatfacto가 optimizer로 갱신하는 게 이것
#
# $l$ 을 **차수(degree, band)** 라 부릅니다. 낮은 $l$ 은 구면 위에서 천천히 변하는 성분(= 저주파),
# 높은 $l$ 은 빠르게 변하는 성분(= 고주파)입니다. 푸리에 급수의 주파수와 정확히 같은 역할입니다.
#
# 각 $l$ 마다 $m = -l \dots l$ 로 $2l+1$ 개가 있으므로, $L$ 차까지 쓰면 계수 개수는
#
# $$
# \sum_{l=0}^{L} (2l+1) = (L+1)^2
# $$
#
# 이게 [`num_sh_bases(degree) = (degree+1)**2`](../nerfstudio/utils/spherical_harmonics.py#L88-L93) 입니다.
#
# ⚠️ 이름 주의: `components_from_spherical_harmonics(degree, ...)` 의 `degree` 는 "차수의 **개수**"가 아니라
# **최대 차수 $L$** 입니다(docstring이 헷갈리게 쓰여 있습니다). `degree=3` → 16개 basis.

# %%
print(f"{'L (최대 차수)':>14s} {'이번 밴드 2L+1':>14s} {'누적 (L+1)^2':>14s}   splatfacto 저장 위치")
_prev = 0
for L in range(MAX_SH_DEGREE + 1):
    n = num_sh_bases(L)
    where = {0: "features_dc  [N, 3]", 1: "features_rest[N, 15, 3] 의 0:3", 2: "  〃  3:8", 3: "  〃  8:15"}.get(L, "(sh_degree=3 기본값 초과)")
    print(f"{L:>14d} {n - _prev:>14d} {n:>14d}   {where}")
    _prev = n

print("\nsplatfacto 기본 sh_degree=3 → 계수 16개 × RGB 3채널 = 가우시안당 48개 실수")
print("  features_dc  : [N,  3]  ← l=0 (계수 1개 × 3채널)")
print("  features_rest: [N, 15, 3] ← l=1,2,3 (계수 15개 × 3채널)")
print("\n왜 나눠 저장하나?  ① l=0은 '평균 색'이라 초기화 방식이 다르고(SfM 포인트 색 주입)")
print("                    ② optimizer 그룹을 분리해 lr을 다르게 주려고 (2.5e-3 vs 그 1/20)")


# %% [markdown]
# ## ② basis를 눈으로 — $Y_{lm}$ 은 어떻게 생겼나
#
# `components_from_spherical_harmonics` 는 하드코딩된 다항식입니다. 예를 들어 $l=0$ 은 상수,
#
# $$Y_{00} = 0.2820947918 = \sqrt{\tfrac{1}{4\pi}} \equiv C_0$$
#
# $l=1$ 은 방향 성분 자체에 비례합니다($0.4886 \cdot y,\ z,\ x$). 아래에서 구면 전체에 대해 그려 봅니다.
#
# 구면을 평면에 펴서(equirectangular: 가로 = 방위각 $\phi$, 세로 = 극각 $\theta$) 각 basis의 값을 색으로 표시합니다.
# **행이 차수 $l$** 이고, 아래로 갈수록 무늬가 잘게 쪼개지는 것 = 고주파라는 뜻입니다.

# %%
def equirect_dirs(n_theta: int = 90, n_phi: int = 180) -> torch.Tensor:
    """구면을 (theta, phi) 격자로 샘플해 단위 방향 [n_theta, n_phi, 3] 을 만든다."""
    theta = torch.linspace(0, np.pi, n_theta)          # 극각: 0=+z(위), pi=-z(아래)
    phi = torch.linspace(-np.pi, np.pi, n_phi)         # 방위각
    t, p = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)], dim=-1)


grid = equirect_dirs()
Y = components_from_spherical_harmonics(3, grid.reshape(-1, 3)).reshape(*grid.shape[:2], -1)  # [90,180,16]
print("basis 값 shape:", tuple(Y.shape), "| l=0 은 상수?", torch.allclose(Y[..., 0], Y[0, 0, 0]))

fig, axes = plt.subplots(4, 7, figsize=(14, 7))
for ax in axes.ravel():
    ax.axis("off")
idx = 0
for L in range(4):
    band = 2 * L + 1
    off = (7 - band) // 2  # 삼각형 모양으로 가운데 정렬
    for m in range(band):
        ax = axes[L, off + m]
        v = Y[..., idx].numpy()
        lim = max(abs(v).max(), 1e-8)
        ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(-180, 180, 180, 0), aspect="auto")
        ax.set_title(f"l={L}, m={m - L}\n[{idx}]", fontsize=8)
        ax.axis("off")
        idx += 1
fig.suptitle("구면조화함수 basis $Y_{lm}$ (equirectangular, 빨강 +/파랑 −)\n아래 행일수록 고주파", fontsize=11)
plt.tight_layout()
plt.show()

# 직교성 확인: 구면 위에서 서로 다른 basis의 내적은 0 (sin(theta) 가중 필요)
w = torch.sin(torch.linspace(0, np.pi, 90))[:, None]                      # 구면 면적 요소
G = torch.einsum("tpi,tpj,tp->ij", Y, Y, w.expand(90, 180)) / (w.sum() * 180 / (4 * np.pi))
print(f"\n직교성(격자 수치적분): 대각 평균 {G.diagonal().mean():.4f} (이론값 1.0), "
      f"비대각 최대 |값| {(G - torch.diag(G.diagonal())).abs().max():.2e} (이론값 0, 나머지는 격자 이산화 오차)")


# %% [markdown]
# ## ③ $C_0$ 와 `RGB2SH` / `SH2RGB` — 왜 0.5를 빼는가
#
# [`RGB2SH`/`SH2RGB`](../nerfstudio/utils/spherical_harmonics.py#L96-L109) 는 두 줄짜리 함수입니다:
#
# $$
# \mathrm{sh}_0 = \frac{\mathrm{rgb} - 0.5}{C_0}, \qquad \mathrm{rgb} = C_0 \cdot \mathrm{sh}_0 + 0.5,
# \qquad C_0 = 0.2820947918
# $$
#
# $C_0$ 로 나누는 건 자명합니다 — $Y_{00} = C_0$ 이므로 $Y_{00} \cdot \mathrm{sh}_0 = \mathrm{rgb} - 0.5$ 가 되게 하려는 것.
#
# **0.5는 어디서 오나?** 래스터라이저 쪽입니다. gsplat은 SH를 평가한 뒤 항상 0.5를 더합니다
# (`gsplat/rendering.py`: `colors = torch.clamp_min(colors + 0.5, 0.0)`).
# 즉 SH가 표현하는 건 색 자체가 아니라 **중간 회색(0.5)으로부터의 편차**입니다.
# 덕분에 계수 0으로 초기화된 가우시안은 회색으로 시작하고, 고차 밴드를 0으로 두면 색에 아무 영향이 없습니다.

# %%
C0 = 0.28209479177387814
print("C0 =", C0, "≈ sqrt(1/(4π)) =", np.sqrt(1 / (4 * np.pi)))
print("Y[0] 과 일치?", np.isclose(C0, Y[0, 0, 0].item()))

rgb = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])  # 빨강 / 중간회색 / 검정
sh0 = RGB2SH(rgb)
print(f"\n{'rgb':>18s} → {'sh_0 (features_dc)':>26s} → {'SH2RGB 왕복':>18s}")
for a, b, c in zip(rgb, sh0, SH2RGB(sh0)):
    print(f"{str(a.tolist()):>18s} → {str([round(v, 3) for v in b.tolist()]):>26s} → {str([round(v, 3) for v in c.tolist()]):>18s}")
print("\n중간회색 0.5 → sh_0 = 0 : SH 계수가 전부 0이면 렌더 결과는 회색")

# sh_degree=0 으로 설정하면 SH를 아예 안 쓰고 sigmoid 경로를 탄다 (splatfacto.py:299-309, 549-553)
print("\nsh_degree=0 일 때: features_dc를 SH가 아니라 sigmoid의 logit으로 해석 → 방향 의존성 없음(램버시안)")


# %% [markdown]
# ## ④ 차수를 올리면 뭐가 좋아지나 — 방향 함수 근사 실험
#
# 백문이 불여일견입니다. **시점 의존 하이라이트**를 흉내낸 목표 함수
#
# $$ f(\mathbf d) = \max(0,\ \mathbf d \cdot \mathbf r)^{\,p} $$
#
# (반사 방향 $\mathbf r$ 주변에 몰린 좁은 로브, Phong 모델의 그것) 을 차수 $L=0,1,2,3,4$ 로 최소제곱 근사합니다.
# 이게 정확히 **가우시안 하나가 자기 색의 방향 의존성을 SH로 저장하는 상황**입니다.

# %%
def fib_sphere(n: int) -> torch.Tensor:
    """구면 위 균등 샘플 (피보나치 격자) [n, 3]"""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n
    r = torch.sqrt(torch.clamp(1 - z * z, min=0))
    a = np.pi * (1 + 5**0.5) * i
    return torch.stack([r * torch.cos(a), r * torch.sin(a), z], dim=-1).float()


dirs = fib_sphere(20000)
r_dir = torch.tensor([0.6, 0.3, 0.74])
r_dir = r_dir / r_dir.norm()
POWER = 8
target = torch.clamp((dirs @ r_dir), min=0) ** POWER  # [20000]

fits, rmses = {}, {}
for L in range(MAX_SH_DEGREE + 1):
    B = components_from_spherical_harmonics(L, dirs)            # [20000, (L+1)^2]
    coef = torch.linalg.lstsq(B, target[:, None]).solution      # [(L+1)^2, 1]  ← 이게 학습으로 찾는 값
    fits[L] = coef.squeeze(-1)
    rmses[L] = (B @ coef).squeeze(-1).sub(target).pow(2).mean().sqrt().item()
    print(f"L={L}: 계수 {num_sh_bases(L):2d}개, RMSE {rmses[L]:.4f}, 상대오차 {rmses[L] / target.std():.1%}")

# 구면 격자에 재구성해서 나란히 보기
flat = grid.reshape(-1, 3)
tgt_img = (torch.clamp(flat @ r_dir, min=0) ** POWER).reshape(90, 180).numpy()
fig, axes = plt.subplots(1, MAX_SH_DEGREE + 2, figsize=(16, 2.8))
axes[0].imshow(tgt_img, cmap="magma", vmin=0, vmax=1, extent=(-180, 180, 180, 0), aspect="auto")
axes[0].set_title(f"목표 $(\\mathbf{{d}}\\cdot\\mathbf{{r}})^{{{POWER}}}$", fontsize=9), axes[0].axis("off")
for L in range(MAX_SH_DEGREE + 1):
    rec = (components_from_spherical_harmonics(L, flat) @ fits[L]).reshape(90, 180).numpy()
    axes[L + 1].imshow(rec, cmap="magma", vmin=0, vmax=1, extent=(-180, 180, 180, 0), aspect="auto")
    axes[L + 1].set_title(f"L={L} ({num_sh_bases(L)}개)\nRMSE {rmses[L]:.3f}", fontsize=9), axes[L + 1].axis("off")
fig.suptitle("차수를 올릴수록 좁은 하이라이트를 표현할 수 있다", fontsize=11)
plt.tight_layout()
plt.show()

print("\n관찰:")
print("  L=0 은 상수 — 방향 의존성을 전혀 표현 못 함 (램버시안 = 어디서 봐도 같은 색)")
print("  L=1 부터 '한쪽이 밝다' 정도가 생기고, L을 올릴수록 로브가 좁아짐")
print("  L=3(splatfacto 기본)에서도 완벽하진 않음 — 거울 반사는 SH로 표현 불가, 그래서 3DGS는 거울/유리에 약하다")


# %% [markdown]
# ## ⑤ splatfacto의 SH 파이프라인 — 저장 → concat → 래스터라이저
#
# 학습 중에는 두 텐서로 나뉘어 있다가, forward 직전에 합쳐집니다 ([splatfacto.py:530](../nerfstudio/models/splatfacto.py#L530)):
#
# ```python
# colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)   # [N, 1, 3] + [N, 15, 3] → [N, 16, 3]
# ...
# sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
# render, alpha, info = rasterization(..., colors=colors_crop, sh_degree=sh_degree_to_use, ...)
# ```
#
# gsplat 안에서 벌어지는 일:
#
# 1. 가우시안마다 방향 $\mathbf d$ = (가우시안 중심 − 카메라 위치) 를 계산
# 2. `spherical_harmonics(sh_degree_to_use, dirs, shs)` — **앞쪽 $(L+1)^2$ 개 계수만** 사용
# 3. `colors = clamp_min(결과 + 0.5, 0)` — ③에서 본 0.5
#
# 아래에서 nerfstudio의 `components_from_spherical_harmonics` 로 손계산한 값과 gsplat CUDA 커널 결과를 대조합니다.
# **결론부터: $l\ge1$ 에서 안 맞습니다.** 두 코드베이스가 서로 다른 부호 규약을 쓰기 때문인데,
# 이걸 파헤치는 게 SH를 제대로 이해했는지 확인하는 좋은 시험대입니다.

# %%
if torch.cuda.is_available():
    from gsplat.cuda._wrapper import spherical_harmonics

    N = 5
    dev = torch.device("cuda")
    d = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1).to(dev)
    shs = torch.randn(N, 16, 3, device=dev) * 0.3
    shs[:, 0] = RGB2SH(torch.rand(N, 3, device=dev))  # l=0 은 실제 색에서

    print(f"{'L':>2s}  {'사용 계수':>8s}  {'gsplat 색 (0번 가우시안, +0.5 후)':>34s}   나이브 손계산 일치")
    for L in range(4):
        gs = spherical_harmonics(L, d, shs)                                   # [N, 3]
        Bl = components_from_spherical_harmonics(L, d.cpu())                  # [N, (L+1)^2]
        manual = torch.einsum("nk,nkc->nc", Bl, shs[:, : num_sh_bases(L)].cpu())
        rgb_gs = torch.clamp_min(gs.cpu() + 0.5, 0.0)
        ok = torch.allclose(gs.cpu(), manual, atol=1e-5)
        print(f"{L:>2d}  {num_sh_bases(L):>8d}  {str([round(v, 4) for v in rgb_gs[0].tolist()]):>34s}   {ok}")
    print("\nl=0 만 맞고 나머지는 전부 불일치 → 상수배 차이인지 확인해 봅니다.")
else:
    print("CUDA 없음 — ⑤ 건너뜀 (gsplat의 spherical_harmonics는 CUDA 커널)")


# %% [markdown]
# ### 왜 안 맞나 — one-hot 프로브로 basis를 직접 추출
#
# gsplat의 SH는 CUDA 커널이라 소스를 읽기 번거롭습니다. 대신 **계수를 one-hot으로 넣으면 basis 값이 그대로 튀어나옵니다**:
# $k$ 번째 계수만 1이고 나머지가 0이면 $\sum_j c_j Y_j = Y_k$ 이니까요.
# 이렇게 뽑은 gsplat의 $Y_k$ 를 nerfstudio 것과 나눠 보면 규약 차이가 드러납니다.

# %%
if torch.cuda.is_available():
    d_probe = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    K = 16
    G = torch.zeros(64, K)
    for k in range(K):
        onehot = torch.zeros(64, K, 3, device=dev)
        onehot[:, k, 0] = 1.0                                   # R 채널만 1 → 반환값의 R이 곧 Y_k
        G[:, k] = spherical_harmonics(3, d_probe.to(dev), onehot)[:, 0].cpu()
    Nb = components_from_spherical_harmonics(3, d_probe)        # [64, 16]

    ratio = G / Nb
    print(f"{'k':>3s} {'l':>2s} {'m':>3s}  {'gsplat / nerfstudio':>20s}  {'방향에 따라 변하나 (std)':>24s}")
    for k in range(K):
        L = int(np.floor(np.sqrt(k)))
        m = k - L * L - L
        print(f"{k:>3d} {L:>2d} {m:>+3d}  {ratio[:, k].mean():>20.4f}  {ratio[:, k].std():>24.1e}")

    sign = torch.tensor([(-1.0) ** (k - int(np.floor(np.sqrt(k))) ** 2 - int(np.floor(np.sqrt(k)))) for k in range(K)])
    print("\n비율이 방향과 무관한 ±1 → 두 basis는 부호만 다르다.")
    print("부호 패턴이 (-1)^m 과 일치?", torch.allclose(ratio.mean(0), sign, atol=1e-4))

    # 부호를 맞추면 완전 일치 — 가우시안 하나의 계수 [16,3] 을 64개 방향에서 평가해 대조
    one = shs[0]                                                                   # [16, 3]
    gs64 = spherical_harmonics(3, d_probe.to(dev), one.unsqueeze(0).expand(64, -1, -1).contiguous()).cpu()
    manual_fixed = torch.einsum("nk,k,kc->nc", Nb, sign, one.cpu())                # (-1)^m 보정 포함
    print("부호 보정 후 gsplat과 일치?", torch.allclose(gs64, manual_fixed, atol=1e-5),
          f"| 최대 오차 {(gs64 - manual_fixed).abs().max():.2e}")


# %% [markdown]
# ### 규약 차이의 정체 — Condon–Shortley phase
#
# 두 basis는 **$(-1)^m$ 만큼 다릅니다.** 물리·화학에서 관례로 붙이는 부호 인자(Condon–Shortley phase)를
# nerfstudio 유틸은 빼고, gsplat(과 3DGS 원본 CUDA 코드)은 넣은 것입니다. 어느 쪽도 틀리지 않았습니다 — 둘 다 정규직교 실수 basis입니다.
#
# 평탄화 인덱스 $k = l^2 + (m+l)$ 에 대해 $k - m = l^2 + l = l(l+1)$ 은 **항상 짝수**이므로,
# $(-1)^m = (-1)^k$ 가 되어 부호가 인덱스 순서대로 $+,-,+,-,\dots$ 로 깔끔하게 번갈아 나타납니다(위 표).
#
# 실무적으로 중요한 점 세 가지:
#
# 1. **학습에는 아무 영향이 없습니다.** 계수가 부호를 그냥 흡수합니다 — $c_k$ 대신 $-c_k$ 를 학습할 뿐.
# 2. **splatfacto는 `components_from_spherical_harmonics` 를 아예 호출하지 않습니다.** SH 평가는 전부 gsplat CUDA 안에서 일어납니다.
#    그 유틸은 nerfstudio의 **NeRF 계열 모델**(방향 인코딩)이 쓰는 것으로, 이름이 같다고 같은 규약이 아닙니다.
# 3. **계수를 코드베이스 간에 옮길 때 터집니다.** PLY export/import, 다른 뷰어로 이식, 논문 재현 시
#    $l\ge1$ 계수의 부호가 뒤집히면 하이라이트가 반대편에 생깁니다. 색이 "대충 맞는데 미묘하게 이상"하면 이걸 의심하세요.

# %%
if torch.cuda.is_available():
    # 뒤쪽 계수를 망가뜨려도 낮은 L에서는 결과가 안 변한다 = features_rest의 grad가 0인 이유
    shs_broken = shs.clone()
    shs_broken[:, 1:] = 1e3
    same = torch.allclose(spherical_harmonics(0, d, shs), spherical_harmonics(0, d, shs_broken))
    print(f"l≥1 계수를 1e3 으로 망가뜨려도 L=0 결과가 동일? {same}")
    print("  → step<1000 에서 features_rest.grad == 0 인 이유. 계수가 계산 그래프에 아예 안 들어감")

    # 부호를 뒤집으면? 학습에는 무해하지만 이식할 때는 치명적
    flipped = shs.clone()
    flipped[:, 1:] *= -1
    a = torch.clamp_min(spherical_harmonics(3, d, shs) + 0.5, 0)[0]
    b = torch.clamp_min(spherical_harmonics(3, d, flipped) + 0.5, 0)[0]
    print(f"\nl≥1 부호만 뒤집었을 때 같은 방향에서의 색: {[round(v, 3) for v in a.tolist()]} vs {[round(v, 3) for v in b.tolist()]}")
    print("  → 규약을 잘못 맞춰 이식하면 이만큼 틀립니다")


# %% [markdown]
# ## ⑥ `sh_degree_interval` — 왜 한 번에 다 풀지 않는가
#
# $$
# \text{sh\_degree\_to\_use}(t) = \min\left(\left\lfloor \frac{t}{\texttt{sh\_degree\_interval}} \right\rfloor,\ \texttt{sh\_degree}\right)
# = \min\left(\left\lfloor \frac{t}{1000} \right\rfloor,\ 3\right)
# $$
#
# 1000스텝마다 밴드를 하나씩 엽니다. 3000스텝 이후에야 16개 계수가 전부 활성화됩니다.
#
# **왜 점진적으로?** 고차 SH는 표현력이 크기 때문에, 기하(위치·크기·회전)가 아직 엉망인 초반에 풀어 주면
# **잘못된 기하를 색으로 덮어버립니다.** 특정 시점에서만 맞는 색을 학습해 floater가 굳어지죠.
# 저주파부터 맞추고 고주파를 나중에 여는 건 coarse-to-fine 전략의 전형입니다.
#
# 부수 효과로 **초반 3000스텝은 `features_rest`가 학습되지 않습니다** — grad가 0이라 Adam이 밟아도
# 값이 안 변합니다(모멘트도 0). train_step 노트북 G단계 표의 `features_rest` 행이 이걸 보여줍니다.

# %%
INTERVAL, MAXDEG = 1000, 3
steps = np.arange(0, 30001)
deg = np.minimum(steps // INTERVAL, MAXDEG)
n_active = (deg + 1) ** 2

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].step(steps, deg, where="post")
ax[0].set(xlabel="step", ylabel="sh_degree_to_use", title="열려 있는 최대 차수", xlim=(0, 6000), yticks=[0, 1, 2, 3])
ax[0].grid(alpha=0.3)
ax[1].step(steps, n_active, where="post", label="활성 계수 $(L+1)^2$")
ax[1].axhline(16, ls="--", c="gray", lw=0.8)
ax[1].set(xlabel="step", ylabel="계수 개수", title="forward에 실제로 쓰이는 계수", xlim=(0, 6000))
ax[1].grid(alpha=0.3), ax[1].legend()
plt.tight_layout()
plt.show()

print(f"{'step 구간':>16s} {'sh_degree_to_use':>17s} {'활성 계수':>10s}  features_rest 중 학습되는 부분")
for lo, hi, L in [(0, 999, 0), (1000, 1999, 1), (2000, 2999, 2), (3000, 29999, 3)]:
    used = num_sh_bases(L)
    part = "없음 (grad=0)" if L == 0 else f"[:, :{used - 1}] ({used - 1}/15)"
    print(f"{f'{lo}–{hi}':>16s} {L:>17d} {used:>10d}  {part}")


# %% [markdown]
# ## ⑦ 학습된 모델의 실제 SH (체크포인트가 있을 때만)
#
# 30k 스텝 완주한 모델에서 실제 계수를 꺼내 봅니다. 두 가지를 확인합니다:
#
# 1. **밴드별 계수 크기** — $l{=}0$ 이 얼마나 지배적인가, 그리고 $l{=}1,2,3$ 사이에는 경향이 있는가?
# 2. **가우시안 하나의 방향별 색** — 실제로 시점에 따라 색이 변하는가?
#
# ⑤에서 본 부호 규약 때문에, 체크포인트의 계수를 `components_from_spherical_harmonics` 로 평가할 때는
# **$(-1)^m$ 보정을 넣어야** gsplat이 렌더한 것과 같은 색이 나옵니다.

# %%
CKPTS = sorted((REPO / "outputs/garden-splatfacto/splatfacto").glob("*/nerfstudio_models/*.ckpt"))
if not CKPTS:
    print("체크포인트 없음 — ⑦ 건너뜀 (`ns-train splatfacto` 완주 후 실행하세요)")
else:
    ckpt = torch.load(CKPTS[-1], map_location="cpu", weights_only=False)
    sd = ckpt["pipeline"]
    dc = next(v for k, v in sd.items() if k.endswith("features_dc"))      # [N, 3]
    rest = next(v for k, v in sd.items() if k.endswith("features_rest"))  # [N, 15, 3]
    print(f"{CKPTS[-1].parent.parent.name} step={ckpt['step']} | features_dc {tuple(dc.shape)} | features_rest {tuple(rest.shape)}")

    # 밴드별 RMS
    bands = {0: dc[:, None, :], 1: rest[:, 0:3], 2: rest[:, 3:8], 3: rest[:, 8:15]}
    print(f"\n{'밴드 l':>7s} {'계수 수':>7s} {'RMS':>10s} {'l=0 대비':>10s}")
    base = bands[0].pow(2).mean().sqrt().item()
    for L, t in bands.items():
        r = t.pow(2).mean().sqrt().item()
        print(f"{L:>7d} {t.shape[1]:>7d} {r:>10.4f} {r / base:>9.1%}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].bar([f"l={L}" for L in bands], [t.pow(2).mean().sqrt().item() for t in bands.values()], color="steelblue")
    ax[0].set(ylabel="계수 RMS", title="밴드별 계수 크기 — 저주파가 지배적")
    ax[0].grid(alpha=0.3, axis="y")

    # ⑤에서 확인한 부호 규약 보정 (-1)^m — 없으면 gsplat이 렌더한 색과 달라진다
    CS = torch.tensor([(-1.0) ** (k - int(np.floor(np.sqrt(k))) ** 2 - int(np.floor(np.sqrt(k)))) for k in range(16)])

    # 시점 의존성이 '눈에 보이는' 가우시안 고르기.
    # rest 에너지만 크면 흰색으로 포화된 놈이 뽑히므로(클램프 후 변화 0), 후보 중 클램프 후 실제 변동이 큰 것을 고른다.
    probe = fib_sphere(64)
    Bp = components_from_spherical_harmonics(3, probe) * CS                 # [64, 16]
    cand = torch.topk(rest.pow(2).sum(dim=(1, 2)), 2000).indices
    coef_c = torch.cat([dc[cand][:, None, :], rest[cand]], dim=1)           # [2000, 16, 3]
    cols = torch.clamp(torch.einsum("pk,nkc->npc", Bp, coef_c) + 0.5, 0.0, 1.0)
    i = int(cand[cols.std(dim=1).mean(dim=-1).argmax()])

    coef = torch.cat([dc[i][None], rest[i]], dim=0)                        # [16, 3]
    B = components_from_spherical_harmonics(3, flat) * CS                  # [16200, 16]
    img = torch.clamp(B @ coef + 0.5, 0.0, 1.0).reshape(90, 180, 3).numpy()
    ax[1].imshow(img, extent=(-180, 180, 180, 0), aspect="auto")
    ax[1].set(title=f"가우시안 #{i} 의 방향별 색 (L=3)", xlabel="방위각 φ", ylabel="극각 θ")
    plt.tight_layout()
    plt.show()

    base_rgb = SH2RGB(dc[i]).clamp(0, 1)
    print(f"\n가우시안 #{i}: l=0 만 쓴 색 {[round(v, 3) for v in base_rgb.tolist()]}  ← 시점 무관 성분")
    print(f"  L=3 방향별 색 범위 R[{img[..., 0].min():.3f}, {img[..., 0].max():.3f}] "
          f"G[{img[..., 1].min():.3f}, {img[..., 1].max():.3f}] B[{img[..., 2].min():.3f}, {img[..., 2].max():.3f}]")
    print("  → 이 폭이 곧 '시점에 따라 색이 얼마나 변하는가' 입니다")
    print(f"\n참고: 전체 {len(dc):,}개 중 rest 에너지 상위 2000개에서 고른 값이라 장면 평균은 이보다 훨씬 작습니다.")


# %% [markdown]
# ## 정리
#
# | 질문 | 답 |
# |---|---|
# | SH 차수 $L$ 이란 | 구면 위 함수의 주파수 상한. $L$ 이 클수록 좁은 하이라이트를 표현 |
# | 계수는 몇 개 | $(L+1)^2$ 개 × RGB 3채널. splatfacto 기본 $L=3$ → 16×3 = 48개/가우시안 |
# | 왜 `features_dc` / `features_rest` 로 나누나 | $l{=}0$ 은 SfM 색으로 초기화하고, **optimizer 그룹을 분리해 lr을 1/20로 낮추기 위해** |
# | $C_0 = 0.282$ 는 | $Y_{00} = \sqrt{1/4\pi}$. `RGB2SH = (rgb-0.5)/C0` |
# | 0.5는 왜 빼나 | 래스터라이저가 렌더 시 항상 0.5를 더함. SH는 **중간 회색으로부터의 편차**를 표현 |
# | `sh_degree_interval` | 1000스텝마다 밴드 하나씩 개방. 기하가 잡히기 전에 고주파 색을 풀면 floater가 굳음 |
# | `features_rest` grad가 0인 이유 | step<1000 이면 `sh_degree_to_use=0` → 고차 계수가 계산 그래프에 아예 안 들어감 |
# | nerfstudio 유틸 vs gsplat | basis가 $(-1)^m$ 만큼 다름(Condon–Shortley). splatfacto는 유틸을 안 쓰고 gsplat CUDA로만 평가. 학습엔 무해하지만 **계수 이식 시 주의** |
# | $L=3$ 의 한계 | 거울 반사 같은 극단적 고주파는 표현 불가 — 3DGS가 유리/금속에 약한 근본 원인 |
#
# 다음: [splatfacto_train_step.py](splatfacto_train_step.py) G단계로 돌아가면 `features_rest` 행의 `grad=0` 이 왜 그런지 이제 보입니다.
