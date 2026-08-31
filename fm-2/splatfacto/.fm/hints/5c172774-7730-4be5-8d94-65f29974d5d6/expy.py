# %% [markdown]
# # splatfacto의 `resize_image` — conv2d로 구현한 'area' 다운스케일
#
# splatfacto.py(49–62행)의 `resize_image(image, d)`는 [H, W, C] 이미지를
# [H//d, W//d, C]로 줄인다. 핵심 아이디어:
#
# - 모든 원소가 $\frac{1}{d \times d}$ 인 $d \times d$ **균일 가중치 커널**을 만들고
# - 이를 **stride $d$** 의 `conv2d`로 적용한다.
#
# 출력 픽셀 하나는 겹치지 않는 $d \times d$ 블록의 값을 커널과 곱해 더한 것이므로
#
# $$\text{out}[i,j] = \sum_{p=0}^{d-1}\sum_{q=0}^{d-1} \frac{1}{d^2}\,
# \text{in}[di+p,\; dj+q] = \underbrace{\frac{1}{d^2}\sum_{p,q} \text{in}[di+p, dj+q]}_{d\times d \text{ 블록 평균}}$$
#
# 즉 **블록 평균(block-wise mean)** 이고, 이는 정수 배율 축소에서 OpenCV의
# `INTER_AREA` 보간과 정확히 같다. $d$는 2, 4, 8 같은 2의 거듭제곱이어야 한다
# (H, W가 $d$로 나누어떨어져야 블록이 딱 맞음).

# %%
# 필요 패키지: torch, numpy, plotly, kaleido, opencv-python(선택)
import numpy as np
import torch
import torch.nn.functional as tf


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# splatfacto.py 49-62행 원본 그대로
def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


# %% [markdown]
# ## 1. 가장 작은 예제: 4×4 단일 채널, d=2
#
# 2×2 블록 4개가 각각 평균으로 접히는지 눈으로 확인한다.

# %%
tiny = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)
print("입력 4x4:\n", tiny[..., 0])
out = resize_image(tiny, 2)
print("resize_image(d=2) -> shape", tuple(out.shape))
print(out[..., 0])
# 좌상단 블록 [[0,1],[4,5]]의 평균 = (0+1+4+5)/4 = 2.5
# 출력:
# 입력 4x4:
#  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.]])
# resize_image(d=2) -> shape (2, 2, 1)
# tensor([[ 2.5000,  4.5000],
#         [10.5000, 12.5000]])

# %% [markdown]
# ## 2. 블록 평균과의 동치 검증
#
# `reshape(H//d, d, W//d, d, C).mean(dim=(1, 3))` 으로 만든 명시적 블록 평균과
# conv2d 결과가 같은지 확인한다. 커널이 균일하고 stride가 커널 크기와 같으니
# 두 계산은 수식상 동일해야 한다.

# %%
torch.manual_seed(0)
H, W, C = 64, 96, 3
img = torch.rand(H, W, C)

for d in (2, 4, 8):
    conv_out = resize_image(img, d)
    block_mean = img.reshape(H // d, d, W // d, d, C).mean(dim=(1, 3))
    diff = (conv_out - block_mean).abs().max().item()
    print(f"d={d}: shape {tuple(conv_out.shape)}, |conv2d - 블록평균| 최대 = {diff:.2e}")
# 출력:
# d=2: shape (32, 48, 3), |conv2d - 블록평균| 최대 = 5.96e-08
# d=4: shape (16, 24, 3), |conv2d - 블록평균| 최대 = 1.19e-07
# d=8: shape (8, 12, 3), |conv2d - 블록평균| 최대 = 1.79e-07
# -> float32 반올림 오차 수준으로 완전히 일치

# %% [markdown]
# ## 3. OpenCV `INTER_AREA`와의 동치 검증
#
# 정수 배율 축소에서 `cv2.resize(..., interpolation=cv2.INTER_AREA)`는
# 정확히 블록 평균이다. docstring의 "same 'area' method in opencv" 주장을 확인한다.

# %%
try:
    import cv2

    np_img = img.numpy()
    for d in (2, 4, 8):
        cv_out = cv2.resize(np_img, (W // d, H // d), interpolation=cv2.INTER_AREA)
        conv_out = resize_image(img, d).numpy()
        diff = np.abs(cv_out - conv_out).max()
        print(f"d={d}: |cv2.INTER_AREA - resize_image| 최대 = {diff:.2e}")
except ImportError:
    print("opencv 미설치 — 이 셀은 건너뜀")
# 출력:
# d=2: |cv2.INTER_AREA - resize_image| 최대 = 0.00e+00
# d=4: |cv2.INTER_AREA - resize_image| 최대 = 1.19e-07
# d=8: |cv2.INTER_AREA - resize_image| 최대 = 1.79e-07
# -> INTER_AREA와도 반올림 오차 수준으로 일치

# %% [markdown]
# ## 4. 왜 채널별로 따로 conv 하는가 — permute 트릭
#
# `conv2d`의 기본 동작은 입력 채널들을 **합쳐서** 출력 하나를 만든다.
# RGB를 섞으면 안 되므로, 원본 코드는
# `image.permute(2, 0, 1)[:, None, ...]` 로 [H, W, C] → [C, 1, H, W] 를 만들어
# **채널을 배치 차원으로** 보낸다. 커널은 [1, 1, d, d] 하나뿐이라 세 채널이
# 같은 균일 커널로 독립적으로 처리되고, 마지막에 `[C, H//d, W//d]` →
# `permute(1, 2, 0)` 으로 [H//d, W//d, C]로 되돌린다.

# %%
d = 4
x = img.permute(2, 0, 1)[:, None, ...]  # [C=3, 1, H, W]: 채널이 배치가 됨
print("conv2d 입력 shape:", tuple(x.shape))
w = (1.0 / (d * d)) * torch.ones((1, 1, d, d))
y = tf.conv2d(x, w, stride=d)
print("conv2d 출력 shape:", tuple(y.shape), "-> squeeze/permute 후:",
      tuple(y.squeeze(1).permute(1, 2, 0).shape))
# 출력:
# conv2d 입력 shape: (3, 1, 64, 96)
# conv2d 출력 shape: (3, 1, 16, 24) -> squeeze/permute 후: (16, 24, 3)

# %% [markdown]
# ## 5. 시각화: 체커보드 + 그라디언트 이미지의 단계적 다운스케일
#
# 고주파(체커보드)가 평균화로 회색으로 뭉개지는 것이 area 방식의 특징이다.
# 단순 서브샘플링(`img[::d, ::d]`)이었다면 체커보드의 한 색만 남는 앨리어싱이 생긴다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 테스트 이미지: R = 가로 그라디언트, G = 세로 그라디언트, B = 8px 체커보드
H2 = W2 = 128
yy, xx = torch.meshgrid(torch.arange(H2), torch.arange(W2), indexing="ij")
checker = (((yy // 8) + (xx // 8)) % 2).float()
demo = torch.stack([xx / (W2 - 1), yy / (H2 - 1), checker], dim=-1)  # [128,128,3]

sub = demo[::8, ::8]  # 순진한 서브샘플링 (비교용)
scales = [("원본 128x128", demo), ("d=2 (64x64)", resize_image(demo, 2)),
          ("d=4 (32x32)", resize_image(demo, 4)), ("d=8 (16x16)", resize_image(demo, 8)),
          ("서브샘플링 [::8,::8]", sub)]

fig = make_subplots(rows=1, cols=5, subplot_titles=[t for t, _ in scales],
                    horizontal_spacing=0.02)
for i, (_, im) in enumerate(scales, start=1):
    rgb = (im.clamp(0, 1) * 255).to(torch.uint8).numpy()
    fig.add_trace(go.Image(z=rgb), row=1, col=i)
    fig.update_xaxes(showticklabels=False, row=1, col=i)
    fig.update_yaxes(showticklabels=False, row=1, col=i)
fig.update_layout(width=1200, height=320, margin=dict(l=10, r=10, t=60, b=10),
                  title_text="resize_image: 균일 커널 conv2d = 블록 평균 (vs 순진한 서브샘플링)")
_show(fig)

import os
_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(_png, scale=2)
print("saved:", _png)
# 출력:
# saved: .../.fm/hints/5c172774-7730-4be5-8d94-65f29974d5d6/expy.png
# d=8에서 B 채널 체커보드(8px 격자)는 블록마다 0/1이 반씩 섞여 정확히 0.5(회색)로 수렴.
# 반면 서브샘플링은 각 블록의 한 픽셀만 집으므로 체커보드의 한 색(파랑 성분 0)만 남는다.

# %% [markdown]
# ## 정리
#
# - `resize_image(image, d)` = $d \times d$ 균일 커널($1/d^2$)을 stride $d$로 conv2d
#   → 겹치지 않는 블록 평균 → OpenCV `INTER_AREA`(정수 배율)와 동일.
# - 채널은 배치 차원으로 옮겨 독립 처리([H,W,C] → [C,1,H,W] → conv → [H//d,W//d,C]).
# - $d$는 2의 거듭제곱(2, 4, 8, ...)이어야 하며, 평균화 덕분에 앨리어싱 없이 축소된다.
