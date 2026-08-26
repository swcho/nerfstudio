# %% [markdown]
# # Splatfacto 학습 과정 해부 — mip-NeRF 360 `garden`
#
# `ns-train splatfacto` 한 줄이 내부적으로 무엇을 하는지, 단계별로 직접 실행하며 확인합니다.
#
# ```
# ① 데이터파서  → ② 데이터매니저 → ③ 가우시안 초기화 → ④ 한 스텝 해부 → ⑤ 미니 학습 루프 → ⑥ 완성 모델과 비교
# ```
#
# **실행 환경**: conda env `nerfstudio`, 커널을 이 env로 선택. GPU 메모리 ~3GB 사용.
# VSCode에서 `# %%` 셀을 `Shift+Enter`로 하나씩 실행하세요.
#
# 참조 코드: [nerfstudio/models/splatfacto.py](../nerfstudio/models/splatfacto.py)

# %%
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # get_viewmat의 torch.compile 비활성화 (노트북에서 재컴파일 잡음 방지)

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
DATA = REPO / "data/mipnerf360/garden"
FINISHED_RUN = REPO / "outputs/garden-splatfacto/splatfacto/2026-08-25_141008"  # 앞서 30k 스텝 완주한 run
assert DATA.exists(), f"{DATA} 없음 — `dm use mipnerf360@2.0.0 --link data/mipnerf360` 먼저"
device = torch.device("cuda")
torch.manual_seed(0)

# matplotlib 한글 폰트 — 설치된 CJK 폰트 중 첫 번째를 사용 (없으면 경고만 뜨고 □로 표시됨)
import matplotlib.font_manager as fm

_ko = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("Noto Sans CJK", "Noto Sans KR", "Nanum", "Malgun"))]
if _ko:
    plt.rcParams["font.family"] = sorted(set(_ko))[0]
plt.rcParams["axes.unicode_minus"] = False
print("repo:", REPO, "| GPU:", torch.cuda.get_device_name(0), "| font:", plt.rcParams["font.family"])


# %% [markdown]
# ## ① 데이터파서 — COLMAP 결과를 nerfstudio 좌표계로
#
# `ColmapDataParser`가 `sparse/0`의 `cameras.bin / images.bin / points3D.bin`을 읽어
# - 카메라 내·외부 파라미터 → `Cameras`
# - 포즈 정규화 (회전·중심·스케일해서 ±1 박스에 맞춤)
# - **SfM 3D 포인트** → `metadata["points3D_xyz"/"points3D_rgb"]`  ← 가우시안 초기값
#
# 우리가 CLI에서 넘긴 플래그(`--colmap-path sparse/0 --downscale-factor 4 --downscale-rounding-mode round`)가 그대로 config입니다.

# %%
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

dataparser_config = ColmapDataParserConfig(
    data=DATA,
    colmap_path=Path("sparse/0"),  # 기본값 colmap/sparse/0 은 ns-process-data 출력 기준
    downscale_factor=4,
    downscale_rounding_mode="round",  # mip-NeRF 360의 images_4는 반올림(5187/4=1296.75→1297)
    load_3D_points=True,
)
dataparser = dataparser_config.setup()
train_out = dataparser.get_dataparser_outputs(split="train")
eval_out = dataparser.get_dataparser_outputs(split="test")

pts = train_out.metadata["points3D_xyz"]
rgb = train_out.metadata["points3D_rgb"]
print(f"train 카메라 {len(train_out.cameras)}장 / eval {len(eval_out.cameras)}장 (eval_interval=8)")
print(f"이미지 해상도 {train_out.cameras.width[0].item()}x{train_out.cameras.height[0].item()}")
print(f"SfM 포인트 {pts.shape[0]:,}개  ← 초기 가우시안 개수")
print(f"scene box: {train_out.scene_box.aabb.tolist()}")

# %%
# 포인트 클라우드 + 카메라 위치 (위에서 내려다본 XY 평면)
cam_pos = train_out.cameras.camera_to_worlds[:, :3, 3].numpy()
sub = torch.randperm(pts.shape[0])[:30000]
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].scatter(pts[sub, 0], pts[sub, 1], c=rgb[sub].numpy() / 255.0, s=0.3)
ax[0].scatter(cam_pos[:, 0], cam_pos[:, 1], c="red", s=8, label="train cams")
ax[0].set_title("SfM points + cameras (top view)"), ax[0].set_aspect("equal"), ax[0].legend()
ax[1].scatter(pts[sub, 0], pts[sub, 2], c=rgb[sub].numpy() / 255.0, s=0.3)
ax[1].scatter(cam_pos[:, 0], cam_pos[:, 2], c="red", s=8)
ax[1].set_title("side view (XZ)"), ax[1].set_aspect("equal")
plt.show()
# 카메라가 테이블 주위를 원형으로 돌고, 포인트는 카메라가 본 곳(테이블·바닥·담벼락)에만 있습니다.
# 하늘/멀리 있는 배경엔 포인트가 없음 → 이 영역은 densification이 채워야 합니다.


# %% [markdown]
# ## ② 데이터매니저 — "ray 배치"가 아니라 "이미지 1장"
#
# NeRF는 픽셀(ray)을 랜덤 샘플링하지만, 가우시안 스플래팅은 래스터화라서 **이미지 전체를 한 번에** 렌더합니다.
# `FullImageDatamanager`는 시작 시 전체 이미지를 undistort → GPU에 uint8로 캐싱하고, 매 스텝 1장을 꺼내줍니다.
#
# 여기부터는 `method_configs["splatfacto"]`(= CLI가 쓰는 프리셋)를 그대로 가져와 데이터파서만 갈아끼웁니다.

# %%
import copy

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig

config = copy.deepcopy(method_configs["splatfacto"])
assert isinstance(config, TrainerConfig)
config.pipeline.datamanager.dataparser = dataparser_config
config.experiment_name = "garden-walkthrough"
config.output_dir = REPO / "outputs"
config.vis = "tensorboard"  # 뷰어 없이
config.logging.local_writer.enable = False  # 콘솔 진행표 끄기
config.set_timestamp()

# Trainer가 하는 setup을 그대로: pipeline(datamanager+model) → optimizers → callbacks
trainer = config.setup(local_rank=0, world_size=1)
trainer.setup(test_mode="test")
pipeline, model, dm = trainer.pipeline, trainer.pipeline.model, trainer.pipeline.datamanager
print(type(dm).__name__, "| cache:", dm.config.cache_images, dm.config.cache_images_type)

# %%
camera, batch = dm.next_train(step=0)
img = batch["image"]
print("camera:", camera.shape, "| image:", tuple(img.shape), img.dtype, "| cam_idx:", camera.metadata["cam_idx"])
plt.figure(figsize=(8, 5)), plt.imshow(img.cpu().numpy()), plt.title("한 스텝의 학습 데이터 = 이미지 1장 + 카메라 1개"), plt.axis("off")
plt.show()


# %% [markdown]
# ## ③ 가우시안 초기화 — `populate_modules()` ([splatfacto.py:189-231](../nerfstudio/models/splatfacto.py#L189-L231))
#
# SfM 포인트 하나가 가우시안 하나가 됩니다. 6개 파라미터 텐서:
#
# | 파라미터 | 초기값 | 저장 공간 | 활성함수(래스터화 직전) |
# |---|---|---|---|
# | `means` | 포인트 위치 | 그대로 | — |
# | `scales` | 3-최근접 이웃 평균거리 | **log** | `exp` |
# | `quats` | 랜덤 회전 | 쿼터니언 | 정규화 |
# | `features_dc` | 포인트 RGB → SH 0차 | SH 계수 | `SH2RGB` |
# | `features_rest` | 0 | SH 1~3차 (15×3) | — |
# | `opacities` | 0.1 | **logit** | `sigmoid` |
#
# log/logit 공간에 두는 이유: optimizer가 제약 없이 움직여도 exp/sigmoid를 거치면 항상 양수/(0,1)이 보장됩니다.

# %%
for name, p in model.gauss_params.items():
    print(f"{name:14s} {tuple(p.shape)}")
print(f"\n총 가우시안 {model.num_points:,}개 (= SfM 포인트 수)")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].hist(torch.exp(model.scales).mean(-1).detach().cpu().numpy(), bins=100, log=True)
ax[0].set_title("초기 scale (3-NN 평균거리): 성긴 곳은 크다")
ax[1].hist(torch.sigmoid(model.opacities).detach().cpu().numpy(), bins=50)
ax[1].set_title("초기 opacity: 전부 0.1 (반투명)")
ax[2].hist(model.features_rest.detach().cpu().numpy().ravel(), bins=50)
ax[2].set_title("초기 SH 고차 계수: 전부 0 (뷰 의존성 없음)")
plt.show()


# %% [markdown]
# ### 3D로 보기 — 가우시안 하나하나를 공간에서
#
# 각 점 = 가우시안 1개. **위치** = `means`, **색** = SH 0차를 RGB로 되돌린 것, **점 크기** = scale 평균(로그 스케일), **투명도** = opacity.
# 마우스로 회전/줌 가능. 브라우저 부담 때문에 5만 개만 샘플링합니다 (`n`으로 조절).
# 빨간 점은 학습 카메라 위치. 학습 후 같은 함수를 다시 호출해서 densification이 어디를 채웠는지 비교합니다.

# %%
import plotly.graph_objects as go

from nerfstudio.utils.spherical_harmonics import SH2RGB


@torch.no_grad()
def plot_gaussians_3d(model, n=50_000, title="", cameras=None, size_range=(1.0, 6.0), seed=0):
    N = model.num_points
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)[: min(n, N)]
    xyz = model.means[idx].cpu()
    col = torch.clamp(SH2RGB(model.features_dc[idx]), 0, 1).cpu()  # SH 0차 → RGB
    opa = torch.sigmoid(model.opacities[idx]).squeeze(-1).cpu()
    sc = torch.exp(model.scales[idx]).mean(-1).cpu()
    # scale을 로그 공간에서 [size_range]로 매핑 (극단값은 1~99 퍼센타일로 클립)
    ls = torch.log(sc)
    lo, hi = torch.quantile(ls, 0.01), torch.quantile(ls, 0.99)
    size = size_range[0] + (torch.clamp(ls, lo, hi) - lo) / (hi - lo + 1e-8) * (size_range[1] - size_range[0])

    rgb_str = [f"rgba({int(r*255)},{int(gg*255)},{int(b*255)},{max(a.item(), 0.05):.2f})" for (r, gg, b), a in zip(col.tolist(), opa)]
    hover = [f"scale={s_:.4f}<br>opacity={a:.3f}" for s_, a in zip(sc.tolist(), opa.tolist())]
    traces = [
        go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker=dict(size=size.numpy(), color=rgb_str, line=dict(width=0)),
            text=hover, hoverinfo="text", name=f"gaussians ({len(idx):,} / {N:,})",
        )
    ]
    if cameras is not None:
        c = cameras.camera_to_worlds[:, :3, 3].cpu()
        traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers", marker=dict(size=3, color="red"), name="train cams"))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{title}  —  {N:,} gaussians", height=700, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor="black"),
        paper_bgcolor="black", font_color="white", legend=dict(x=0, y=1),
    )
    return fig


plot_gaussians_3d(model, title="step 0 — SfM 포인트 그대로", cameras=dm.train_dataset.cameras).show()
# 보이는 것: 테이블·바닥·담벼락처럼 카메라가 잘 본 곳만 점이 있고, 하늘/먼 배경은 비어 있음.
# 점 크기가 큰 것 = 이웃이 멀어서(성긴 영역) 초기 scale이 큰 가우시안 — 주로 장면 외곽.


# %% [markdown]
# ### 타원체로 보기 — "3-NN 평균거리"가 정확히 무엇인가
#
# 가우시안 하나의 모양은 `scales`(3축 표준편차) + `quats`(회전)로 정해지는 **타원체**입니다.
# 초기화([splatfacto.py:194-197](../nerfstudio/models/splatfacto.py#L194-L197))는
#
# ```python
# distances, _ = k_nearest_sklearn(means.data, 3)   # 각 점에서 가장 가까운 다른 점 3개까지의 거리
# avg_dist = distances.mean(dim=-1, keepdim=True)   # 그 3개 거리의 평균
# scales = log(avg_dist.repeat(1, 3))               # 세 축 모두 같은 값 → 초기엔 '구'
# ```
#
# 즉 **"내 주변 3개 이웃까지의 평균 거리"를 반지름으로 하는 구**로 시작합니다.
# 점이 빽빽한 곳(테이블 상판)은 작은 구, 성긴 곳(장면 외곽)은 큰 구가 되어, 처음부터 대략 표면을 빈틈없이 덮게 하려는 휴리스틱입니다.
# 아래에서 (1) 한 가우시안과 그 3개 이웃, (2) 한 영역의 가우시안 300개를 실제 타원체 메시로 그립니다.

# %%
from nerfstudio.utils.math import k_nearest_sklearn


def quat_to_rotmat(q):
    """gsplat 규약 (w, x, y, z) → 회전행렬 [N,3,3]"""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _unit_sphere(res=10):
    """위도-경도 격자 구 메시. returns verts [V,3], faces [F,3]"""
    th = torch.linspace(0, np.pi, res)
    ph = torch.linspace(0, 2 * np.pi, 2 * res)
    T, P = torch.meshgrid(th, ph, indexing="ij")
    v = torch.stack([torch.sin(T) * torch.cos(P), torch.sin(T) * torch.sin(P), torch.cos(T)], -1).reshape(-1, 3)
    f = []
    W = 2 * res
    for i in range(res - 1):
        for j in range(W):
            a, b = i * W + j, i * W + (j + 1) % W
            c, d = (i + 1) * W + j, (i + 1) * W + (j + 1) % W
            f += [[a, b, c], [b, d, c]]
    return v, torch.tensor(f)


@torch.no_grad()
def ellipsoid_trace(model, idx, n_sigma=1.0, res=10, opacity=0.6, name="gaussians"):
    """idx에 해당하는 가우시안들을 n_sigma 타원체 메시 하나(Mesh3d)로 합쳐서 반환"""
    mu = model.means[idx].cpu()
    S = torch.exp(model.scales[idx]).cpu() * n_sigma          # [N,3] 축별 반지름
    R = quat_to_rotmat(model.quats[idx].cpu())                # [N,3,3]
    col = torch.clamp(SH2RGB(model.features_dc[idx]), 0, 1).cpu()
    v0, f0 = _unit_sphere(res)
    V = v0.shape[0]
    # x = R @ diag(S) @ unit + mu
    verts = torch.einsum("nij,nvj->nvi", R, v0[None] * S[:, None, :]) + mu[:, None, :]   # [N,V,3]
    faces = f0[None] + (torch.arange(len(idx)) * V)[:, None, None]                        # [N,F,3]
    verts, faces = verts.reshape(-1, 3), faces.reshape(-1, 3)
    vcol = (col[:, None, :].expand(-1, V, -1).reshape(-1, 3) * 255).int()
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=[f"rgb({r},{g},{b})" for r, g, b in vcol.tolist()],
        opacity=opacity, flatshading=False, name=name, showlegend=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
    )


def _dark_scene(fig, title, height=650):
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor="black"),
        paper_bgcolor="black", font_color="white", legend=dict(x=0, y=1),
    )
    return fig


# 장면 중심(≈ 테이블) 근처에서 가우시안 K개를 고른다 — 이후 학습 후에도 같은 CENTER로 다시 그려 비교
CENTER = model.means.detach().median(dim=0).values
K_REGION = 300


@torch.no_grad()
def region_idx(model, center=CENTER, k=K_REGION):
    d = (model.means - center.to(model.means.device)).norm(dim=-1)
    return d.topk(k, largest=False).indices


# %%
# (1) 가우시안 하나와 그 3개 이웃 — 초기 반지름 = 3개 거리의 평균 임을 직접 확인
with torch.no_grad():
    means_cpu = model.means.detach().cpu()
    ridx = region_idx(model)
    pick = ridx[0].item()                                      # 중심에 가장 가까운 가우시안 하나
    d_all = (means_cpu - means_cpu[pick]).norm(dim=-1)
    nn_d, nn_i = d_all.topk(4, largest=False)                  # 자기 자신(0) + 이웃 3개
    nn_d, nn_i = nn_d[1:], nn_i[1:]
    r_init = torch.exp(model.scales[pick]).cpu()

print(f"가우시안 #{pick}")
print(f"  3개 이웃까지 거리: {[round(x, 5) for x in nn_d.tolist()]}")
print(f"  평균 = {nn_d.mean():.5f}")
print(f"  exp(scales)     = {[round(x, 5) for x in r_init.tolist()]}   ← 세 축이 모두 평균거리와 같다 (구)")

# 검증: 코드가 실제로 쓰는 함수로도 같은 값이 나오는지 (전체 138k에 대해 sklearn KNN)
dist_all, _ = k_nearest_sklearn(means_cpu, 3)
print(f"  k_nearest_sklearn(means, 3).mean() for #{pick} = {dist_all[pick].mean():.5f}")

nbr = torch.cat([torch.tensor([pick]), nn_i])
traces = [ellipsoid_trace(model, nbr[:1], n_sigma=1.0, res=14, opacity=0.35, name="선택한 가우시안 (1σ 구)")]
for i in nn_i.tolist():  # 이웃으로 가는 선
    a, b = means_cpu[pick], means_cpu[i]
    traces.append(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines", line=dict(color="yellow", width=4), showlegend=False))
traces.append(go.Scatter3d(x=means_cpu[nn_i, 0], y=means_cpu[nn_i, 1], z=means_cpu[nn_i, 2], mode="markers", marker=dict(size=5, color="yellow"), name="3 nearest neighbors"))
# 주변 맥락: 그 근처의 다른 가우시안들 (점)
near = d_all.topk(200, largest=False).indices
traces.append(go.Scatter3d(x=means_cpu[near, 0], y=means_cpu[near, 1], z=means_cpu[near, 2], mode="markers", marker=dict(size=2, color="gray"), name="주변 가우시안 200개"))
_dark_scene(go.Figure(traces), f"3-NN 평균거리 = 초기 반지름  (avg {nn_d.mean():.4f} = exp(scale) {r_init[0]:.4f})").show()
# 구의 표면이 노란 이웃 3개 중 '가운데쯤'을 지납니다 — 가장 가까운 이웃은 구 안에, 가장 먼 이웃은 구 밖에.

# %%
# (2) 한 영역의 가우시안 300개를 실제 타원체(1σ)로. step 0에서는 크기만 다른 구들입니다.
ridx = region_idx(model)
with torch.no_grad():
    ratio0 = (torch.exp(model.scales[ridx]).amax(-1) / torch.exp(model.scales[ridx]).amin(-1)).cpu()
print(f"영역 내 scale 비율(max/min): 평균 {ratio0.mean():.3f}  ← 1.0이면 완전한 구")
_dark_scene(go.Figure([ellipsoid_trace(model, ridx, n_sigma=1.0, res=8)]), f"step 0 — 테이블 근처 {K_REGION}개, 1σ 타원체 (전부 구)").show()


# %%
# 렌더 헬퍼 — 학습 중간에 같은 카메라로 계속 찍어서 비교합니다.
def psnr(a, b):
    return (-10 * torch.log10(((a - b) ** 2).mean())).item()


@torch.no_grad()
def render(camera):
    was_training = model.training
    model.eval()  # eval 모드: downscale 없이 풀해상도, 배경은 고정색
    out = model.get_outputs_for_camera(camera)
    if was_training:
        model.train()
    return {k: v.cpu() for k, v in out.items() if torch.is_tensor(v)}


# 눈으로 확인할 고정 카메라: eval 셋의 첫 장 (학습에 안 쓰인 시점)
eval_cam = dm.eval_dataset.cameras[0:1].to(device)
eval_gt = dm.cached_eval[0]["image"].float().cpu() / 255.0

out0 = render(eval_cam)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].imshow(eval_gt), ax[0].set_title("GT (eval 카메라)")
ax[1].imshow(out0["rgb"]), ax[1].set_title(f"step 0 렌더 — 색칠된 SfM 점구름  PSNR {psnr(out0['rgb'], eval_gt):.2f}")
ax[2].imshow(out0["accumulation"].squeeze(), cmap="gray", vmin=0, vmax=1), ax[2].set_title("accumulation: 가우시안이 덮은 영역")
[a.axis("off") for a in ax]
plt.show()
# 학습 전에도 대충 알아볼 수 있는 이유: 위치와 색이 SfM에서 왔기 때문. 하지만 구멍이 많고(검은 영역) 뿌옇습니다.


# %% [markdown]
# ## ④ 한 스텝 해부 — forward / loss / backward / densify 신호
#
# `Trainer.train_iteration()` 한 번은 다음과 같습니다 ([trainer.py:246-273](../nerfstudio/engine/trainer.py#L246-L273)):
#
# ```
# BEFORE 콜백 (model.step_cb: step 갱신 + optimizers 주입)
#   → get_outputs(camera)   : 투영 + 래스터화 + 배경 합성          splatfacto.py:485
#   → get_loss_dict()       : 0.8·L1 + 0.2·(1-SSIM)                 splatfacto.py:652
#   → backward → optimizer.step (6개 그룹)
# AFTER 콜백  (strategy.step_post_backward: 100스텝마다 분할/복제/컬링)  splatfacto.py:365
# ```

# %%
from nerfstudio.engine.callbacks import TrainingCallbackLocation

pipeline.train()
# BEFORE 콜백 = model.step_cb (L407): step 갱신 + optimizers/schedulers를 모델에 주입.
# densification 전략이 가우시안을 늘릴 때 Adam 상태도 함께 늘려야 하므로 모델이 optimizer를 알아야 합니다.
for cb in trainer.callbacks:
    cb.run_callback_at_location(0, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
camera, batch = dm.next_train(step=0)

# --- forward ---
outputs = model.get_outputs(camera)
print("현재 downscale factor:", model._get_downscale_factor(), "→ 학습 해상도", tuple(outputs["rgb"].shape[:2]))
print("SH 사용 차수:", min(model.step // model.config.sh_degree_interval, model.config.sh_degree), "(1000스텝마다 +1)")
print("배경색(랜덤):", outputs["background"].tolist())

# 래스터라이저가 돌려준 부가 정보 — densification의 원료
info = model.info
print("\nrasterization info:", {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in info.items() if k in ("radii", "means2d", "width", "height", "n_cameras")})
visible = (info["radii"] > 0).sum().item()
print(f"이 시점에서 보이는 가우시안: {visible:,} / {model.num_points:,}")

# --- loss ---
loss_dict = model.get_loss_dict(outputs, batch)
gt_small = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
print("\nloss:", {k: round(v.item(), 4) for k, v in loss_dict.items()})
print("   = 0.8·L1 + 0.2·(1-SSIM)  →  L1 =", round(torch.abs(gt_small - outputs["rgb"]).mean().item(), 4))

# --- backward ---
loss = torch.stack(list(loss_dict.values())).sum()
loss.backward()
g = info["means2d"].absgrad if getattr(info["means2d"], "absgrad", None) is not None else info["means2d"].grad
print("\nmeans2d gradient (화면공간 위치 그래디언트) shape:", tuple(g.shape))
print("  → 이 값의 norm이 densify_grad_thresh(%.4f)를 넘는 가우시안이 분할/복제 후보" % model.config.densify_grad_thresh)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].imshow(torch.clamp(outputs["rgb"].detach().cpu(), 0, 1)), ax[0].set_title("학습 해상도 렌더 (1/4, 랜덤 배경)"), ax[0].axis("off")
gn = g[0].norm(dim=-1).detach().cpu().numpy()
ax[1].hist(gn[gn > 0], bins=100, log=True), ax[1].axvline(model.config.densify_grad_thresh, color="r", ls="--", label="densify_grad_thresh")
ax[1].set_title("가우시안별 2D 위치 그래디언트 norm"), ax[1].legend()
plt.show()
model.zero_grad()
trainer.optimizers.zero_grad_all()


# %% [markdown]
# ## ⑤ 미니 학습 루프 — 3,000 스텝을 실제 Trainer 방식으로
#
# `trainer.train()`의 루프 본체를 그대로 옮겼습니다. 이 구간에서 일어나는 일:
# - step 0~500: 워밍업 (densify 없음)
# - step 500~: 100스텝마다 분할/복제/컬링 → 가우시안 수 급증
# - step 1000, 2000: SH 1차, 2차 켜짐
# - step 3000: 해상도 1/4 → 1/2 전환 + **첫 알파 리셋** (`reset_alpha_every`×`refine_every` = 30×100)
#   - 리셋 직후 loss가 0.04 → 0.10으로 튀고, `pause_refine_after_reset`(161+100 스텝) 동안 densify가 멈춥니다
#   - step 3300에서 재개되며 **대량 컬링**(~5.7만 개)이 일어납니다 — 리셋 후에도 불투명도를 회복하지 못한 가우시안이 정리되는 것
#
# gsplat이 `verbose=True`라 100스텝마다 `duplicated / split / pruned` 개수를 출력합니다. 이 숫자를 지켜보세요.
# RTX 3090에서 약 30~60초.

# %%
import time

SNAP_AT = [0, 200, 500, 800, 1000, 1500, 2000, 2500, 3000, 3100, 3300]
N_STEPS = max(SNAP_AT) + 1
history = {"step": [], "loss": [], "n_gauss": [], "downscale": []}
snaps = {}

t0 = time.time()
for step in range(N_STEPS):
    if step in SNAP_AT:
        snaps[step] = render(eval_cam)["rgb"]
    trainer.step = step
    pipeline.train()
    for cb in trainer.callbacks:  # BEFORE: model.step_cb → self.step = step
        cb.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
    loss, loss_dict, metrics = trainer.train_iteration(step)
    for cb in trainer.callbacks:  # AFTER: strategy.step_post_backward → densify/cull
        cb.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
    history["step"].append(step)
    history["loss"].append(loss.item())
    history["n_gauss"].append(model.num_points)
    history["downscale"].append(model._get_downscale_factor())
    if step % 500 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f} | gaussians {model.num_points:>9,} | 1/{model._get_downscale_factor()} res | {time.time()-t0:5.1f}s")
print(f"done in {time.time()-t0:.1f}s")

# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
s = np.array(history["step"])
ax[0].plot(s, history["n_gauss"]), ax[0].axvline(500, c="gray", ls=":", label="warmup end"), ax[0].set_title("가우시안 개수"), ax[0].legend()
ax[1].plot(s, history["loss"], lw=0.5), ax[1].set_yscale("log"), ax[1].set_title("train loss (3000에서 알파 리셋 스파이크)")
for x in (1000, 2000, 3000):
    ax[1].axvline(x, c="orange", ls=":", lw=0.8)
ax[2].step(s, history["downscale"]), ax[2].set_yticks([1, 2, 4]), ax[2].set_title("학습 해상도 downscale (3000에서 4→2)")
plt.show()

# %%
cols = 4
rows = int(np.ceil(len(snaps) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.3 * rows))
for a, (st, im) in zip(axes.ravel(), snaps.items()):
    a.imshow(im), a.set_title(f"step {st}  PSNR {psnr(im, eval_gt):.2f}"), a.axis("off")
for a in axes.ravel()[len(snaps):]:
    a.axis("off")
plt.suptitle("eval 카메라(학습에 안 쓰임) 시점에서 본 학습 진행", y=1.0)
plt.tight_layout(), plt.show()
# 0→500: 점구름이 뭉개져서 뿌연 상 / 500→1000: densify 시작, 잔디·잎이 채워짐 / 3000→3100: 알파 리셋 직후 일시적으로 흐려짐

# %%
# densification이 "어디에" 가우시안을 추가했는가 — 초기 vs 현재 위치 밀도 비교
now = model.means.detach().cpu()
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].hexbin(pts[:, 0], pts[:, 1], gridsize=120, bins="log", cmap="magma"), ax[0].set_title(f"step 0: {pts.shape[0]:,}개"), ax[0].set_aspect("equal")
ax[1].hexbin(now[:, 0], now[:, 1], gridsize=120, bins="log", cmap="magma"), ax[1].set_title(f"step {N_STEPS-1}: {now.shape[0]:,}개"), ax[1].set_aspect("equal")
plt.show()

# %%
# 같은 시드로 샘플링한 3D 뷰 — step 0 그림과 회전시켜 비교해 보세요.
# 새로 생긴 작은 점들(분할 결과)이 잔디·나뭇잎 영역에 몰려 있고, 알파 리셋 직후라 투명한(흐린) 점이 많습니다.
plot_gaussians_3d(model, title=f"step {N_STEPS-1} — densification 후", cameras=dm.train_dataset.cameras).show()

# %%
# 같은 CENTER 주변 300개를 타원체로 — 이제 구가 아닙니다.
# 테이블 상판처럼 평평한 곳은 납작한 디스크, 잔디/가장자리는 길쭉한 바늘 모양으로 바뀝니다.
# `use_scale_regularization`(PhysGaussian)이 억제하려는 게 바로 이 max/min 비율이 큰 '스파이크'입니다.
ridx_t = region_idx(model)
with torch.no_grad():
    sc_t = torch.exp(model.scales[ridx_t])
    ratio_t = (sc_t.amax(-1) / sc_t.amin(-1)).cpu()
print(f"영역 내 scale 비율(max/min): 평균 {ratio_t.mean():.2f}, 최대 {ratio_t.max():.1f}  (step 0은 1.0)")
_dark_scene(go.Figure([ellipsoid_trace(model, ridx_t, n_sigma=1.0, res=8)]), f"step {N_STEPS-1} — 같은 영역 {K_REGION}개, 1σ 타원체").show()

# %%
# 파라미터 분포가 어떻게 바뀌었나
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].hist(torch.sigmoid(model.opacities).detach().cpu().numpy(), bins=50, log=True), ax[0].set_title("opacity — 0.1에서 양극화 (리셋 직후라 낮은 쪽 많음)")
ax[1].hist(torch.exp(model.scales).mean(-1).detach().cpu().numpy(), bins=100, log=True), ax[1].set_title("scale — 분할로 작은 가우시안 증가")
ratio = (torch.exp(model.scales).amax(-1) / torch.exp(model.scales).amin(-1)).detach().cpu().numpy()
ax[2].hist(np.clip(ratio, 0, 50), bins=100, log=True), ax[2].axvline(model.config.max_gauss_ratio, c="r", ls="--", label="max_gauss_ratio")
ax[2].set_title("scale 비율(max/min) — 길쭉한 스파이크 가우시안"), ax[2].legend()
plt.show()


# %% [markdown]
# ## ⑥ 30k 스텝 완주 모델과 비교
#
# 위에서 3k 스텝만 돌렸으니, 앞서 CLI로 18분 동안 완주한 체크포인트(`step-000029999.ckpt`)를 불러와 같은 카메라로 비교합니다.
# 또 그 run의 TensorBoard 기록에서 전체 곡선을 읽어 미니 루프가 전체의 어느 구간이었는지 확인합니다.

# %%
# ns-eval / ns-viewer / ns-export가 쓰는 공용 로더. config.yml → pipeline 재구성 → 최신 ckpt 로드.
# SplatfactoModel.load_state_dict(L343)가 가우시안 개수에 맞춰 파라미터를 리사이즈해 줍니다.
from nerfstudio.utils.eval_utils import eval_setup

_, final_pipeline, ckpt_path, ckpt_step = eval_setup(FINISHED_RUN / "config.yml", test_mode="inference")
final_model = final_pipeline.model
print("체크포인트:", ckpt_path.name, "| step", ckpt_step, "| 가우시안", f"{final_model.num_points:,}")

final_model.eval()
with torch.no_grad():
    out_final = {k: v.cpu() for k, v in final_model.get_outputs_for_camera(eval_cam).items() if torch.is_tensor(v)}

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].imshow(eval_gt), ax[0].set_title("GT")
ax[1].imshow(snaps[max(snaps)]), ax[1].set_title(f"step {max(snaps)} (이 노트북)  PSNR {psnr(snaps[max(snaps)], eval_gt):.2f}")
ax[2].imshow(out_final["rgb"]), ax[2].set_title(f"step 30000 (CLI 완주)  PSNR {psnr(out_final['rgb'], eval_gt):.2f}")
[a.axis("off") for a in ax]
plt.show()

# %%
# 30k 완주 모델의 3D 뷰 — 1.17M개 중 5만 개 샘플. 하늘/배경까지 채워져 있고 opacity가 양극화된 것이 보입니다.
plot_gaussians_3d(final_model, title="step 30000 — 완주", cameras=dm.train_dataset.cameras).show()

# %%
# 완주 run의 TensorBoard 스칼라 읽기 (tensorboard 서버 없이 이벤트 파일 직접 파싱)
import collections
import struct

from tensorboard.compat.proto import event_pb2

series = collections.defaultdict(list)
for f in FINISHED_RUN.glob("events.out.tfevents.*"):
    data = f.read_bytes()
    i = 0
    while i < len(data):
        (ln,) = struct.unpack("<Q", data[i : i + 8])
        i += 12
        ev = event_pb2.Event()
        ev.ParseFromString(data[i : i + ln])
        i += ln + 4
        for v in ev.summary.value:
            if v.HasField("simple_value"):
                series[v.tag].append((ev.step, v.simple_value))


def S(tag):
    a = np.array(sorted(series[tag]))
    return a[:, 0], a[:, 1]


fig, ax = plt.subplots(1, 3, figsize=(18, 4))
x, y = S("Train Metrics Dict/gaussian_count")
ax[0].plot(x, y), ax[0].axvspan(0, N_STEPS, alpha=0.15, color="orange", label="이 노트북이 돌린 구간")
ax[0].axvline(15000, c="r", ls="--", label="stop_split_at"), ax[0].set_title("가우시안 개수 (30k)"), ax[0].legend()
x, y = S("Eval Images Metrics Dict (all images)/psnr")
ax[1].plot(x, y, "o-"), ax[1].set_title("eval PSNR (24장 평균)"), ax[1].axvspan(0, N_STEPS, alpha=0.15, color="orange")
x, y = S("Train Loss")
ax[2].plot(x, y, lw=0.6), ax[2].set_title("train loss — 3000마다 알파 리셋 스파이크"), ax[2].axvspan(0, N_STEPS, alpha=0.15, color="orange")
for r in range(3000, 15001, 3000):
    ax[2].axvline(r, c="gray", ls=":", lw=0.6)
plt.show()

# %% [markdown]
# ## 정리
#
# | 단계 | 코드 | 핵심 |
# |---|---|---|
# | 초기화 | `populate_modules` L189 | SfM 포인트 138k개 = 초기 가우시안. scale은 이웃 거리, opacity 0.1 |
# | forward | `get_outputs` L485 | 해상도 스케줄(`2^(2-step//3000)`), SH 차수 스케줄(`step//1000`), `rasterization()` 한 번 |
# | loss | `get_loss_dict` L652 | `0.8·L1 + 0.2·(1-SSIM)` — 이게 전부 |
# | densify | `step_post_backward` L365 → gsplat `DefaultStrategy` | 100스텝마다 grad 큰 놈 분할/복제, 투명한 놈 삭제, 3000마다 알파 리셋 |
# | 종료 | `stop_split_at=15000` | 이후 15k 스텝은 개수 고정, 파라미터만 미세조정 |
#
# **다음 실험 아이디어** — 이 스크립트에서 `config.pipeline.model.<flag>`를 바꿔 ⑤를 다시 돌려보세요:
# - `densify_grad_thresh=0.0002` → 가우시안이 훨씬 빨리, 많이 늘어남 (메모리 주의)
# - `cull_alpha_thresh=0.005` → 반투명 가우시안 생존 → 품질↑ 용량↑
# - `num_downscales=0` → 처음부터 풀해상도. 초반 loss 곡선이 어떻게 달라지는지
# - `strategy="mcmc", max_gs_num=300_000` → 개수 상한이 있는 MCMC 전략

# %%
# GPU 정리
del trainer, pipeline, model, final_model, dm
torch.cuda.empty_cache()
print("done")
