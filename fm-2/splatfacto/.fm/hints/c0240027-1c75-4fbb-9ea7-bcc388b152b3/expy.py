# %% [markdown]
# # Splatfacto: 최종 RGB 합성 — `rgb = render + (1 - alpha) * background`
#
# `SplatfactoModel.get_outputs()`에서 gsplat의 `rasterization()`은 두 가지를 돌려준다:
#
# - `render`: 가우시안들을 앞에서 뒤로(front-to-back) 알파 블렌딩한 **premultiplied** 색.
#   배경이 아직 섞이지 않은 상태라서, 아무 가우시안도 덮지 않은 픽셀은 그냥 0이다.
# - `alpha`: 픽셀별 누적 불투명도(accumulation). $\alpha \in [0, 1]$.
#
# 최종 RGB는 알파 합성(over 연산)으로 만든다 (splatfacto.py 582–584행):
#
# ```python
# background = self._get_background_color()
# rgb = render[:, ..., :3] + (1 - alpha) * background
# rgb = torch.clamp(rgb, 0.0, 1.0)
# ```
#
# 수식으로 쓰면 픽셀마다
#
# $$\mathrm{rgb} = \underbrace{\sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j)}_{\text{render (premultiplied)}} \;+\; \Big(1-\underbrace{\sum_i \alpha_i \prod_{j<i}(1-\alpha_j)}_{\alpha}\Big)\cdot \mathrm{bg}$$
#
# 즉 **가우시안이 커버하지 못한(alpha가 낮은) 영역은 배경색이 그대로 비쳐 보인다.**
# 학습 시 `background_color="random"`이면 매 스텝 다른 배경이 들어가서,
# 모델이 "빈 공간을 반투명 가우시안으로 대충 채우는" 꼼수를 못 쓰게 만든다.

# %%
# 필요 패키지: numpy, plotly, kaleido(정적 이미지 저장용)
import os

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


HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 장난감 렌더러: 가우시안 2D 스플랫을 front-to-back 합성
#
# 실제 rasterization을 흉내 내서, 2D 가우시안 몇 개를 앞에서 뒤로 합성해
# `render`(premultiplied 색)와 `alpha`(누적 불투명도)를 만든다.
#
# $$T_0 = 1,\quad \mathrm{render} \mathrel{+}= c_i\,\alpha_i\,T,\quad T \mathrel{*}= (1-\alpha_i)$$
#
# 최종 $\alpha = 1 - T$ (T는 남은 투과율).

# %%
H, W = 96, 128
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

# (중심x, 중심y, 반경, 불투명도, RGB색) — 앞에서 뒤 순서
gaussians = [
    (45, 40, 14, 0.95, np.array([0.90, 0.20, 0.20])),  # 빨강, 거의 불투명
    (70, 50, 18, 0.60, np.array([0.20, 0.55, 0.95])),  # 파랑, 반투명
    (90, 35, 10, 0.80, np.array([0.25, 0.80, 0.30])),  # 초록
]

render = np.zeros((H, W, 3))  # premultiplied 색 (배경 미포함)
T = np.ones((H, W, 1))  # 투과율(transmittance)

for cx, cy, r, opac, color in gaussians:
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))
    a_i = (opac * g)[..., None]  # 픽셀별 알파 기여
    render += color * a_i * T  # c_i * alpha_i * prod(1-alpha_j)
    T *= 1 - a_i

alpha = 1 - T  # 누적 불투명도 = rasterization이 돌려주는 alpha

print("render(premultiplied) 범위:", render.min().round(3), "~", render.max().round(3))
print("alpha 범위:", alpha.min().round(3), "~", alpha.max().round(3))
print("커버 안 된 구석 픽셀 (0,0): render =", render[0, 0].round(4), ", alpha =", alpha[0, 0, 0].round(4))
# 출력: render(premultiplied) 범위: 0.0 ~ 0.855
# 출력: alpha 범위: 0.0 ~ 0.95
# 출력: 커버 안 된 구석 픽셀 (0,0): render = [0. 0. 0.] , alpha = 0.0

# %% [markdown]
# 커버되지 않은 픽셀은 `render = (0,0,0)`, `alpha = 0` — 배경을 섞기 전에는 그냥 검다.

# %% [markdown]
# ## 2. splatfacto의 합성 공식 적용: 배경색별 비교
#
# `_get_background_color()`가 주는 배경(black / white / random)에 대해
# $\mathrm{rgb} = \mathrm{render} + (1-\alpha)\cdot\mathrm{bg}$ 를 적용하고 [0, 1]로 clamp한다.

# %%
def composite(render, alpha, background):
    """splatfacto.py 583–584행과 동일한 연산."""
    rgb = render + (1 - alpha) * background  # background: (3,) — 브로드캐스트
    return np.clip(rgb, 0.0, 1.0)  # torch.clamp(rgb, 0.0, 1.0)


backgrounds = {
    "black": np.zeros(3),
    "white": np.ones(3),
    "random": rng.random(3),  # 학습 기본값: 매 스텝 랜덤
}

composited = {name: composite(render, alpha, bg) for name, bg in backgrounds.items()}

for name, bg in backgrounds.items():
    out = composited[name]
    print(f"bg={name} {bg.round(3)}: 구석 픽셀(미커버) rgb = {out[0, 0].round(3)}")
# 출력: bg=black [0. 0. 0.]: 구석 픽셀(미커버) rgb = [0. 0. 0.]
# 출력: bg=white [1. 1. 1.]: 구석 픽셀(미커버) rgb = [1. 1. 1.]
# 출력: bg=random [0.637 0.27  0.041]: 구석 픽셀(미커버) rgb = [0.637 0.27  0.041]

# %% [markdown]
# 미커버 픽셀($\alpha=0$)의 최종색이 **정확히 배경색**이 되는 것을 확인할 수 있다.
# 반대로 $\alpha \approx 1$인 픽셀은 배경이 거의 안 섞인다.

# %% [markdown]
# ## 3. 픽셀 하나로 보는 수치 예제
#
# 반투명 파랑 가우시안의 가장자리 픽셀($\alpha = 0.3$)을 예로 들면:

# %%
r_pix = np.array([0.06, 0.165, 0.285])  # premultiplied: c * alpha = (0.2,0.55,0.95)*0.3
a_pix = 0.3
for name, bg in backgrounds.items():
    out = np.clip(r_pix + (1 - a_pix) * bg, 0, 1)
    print(f"alpha=0.3, bg={name}: rgb = {out.round(3)}  (배경 기여 {(1 - a_pix) * 100:.0f}%)")
# 출력: alpha=0.3, bg=black: rgb = [0.06  0.165 0.285]  (배경 기여 70%)
# 출력: alpha=0.3, bg=white: rgb = [0.76  0.865 0.985]  (배경 기여 70%)
# 출력: alpha=0.3, bg=random: rgb = [0.506 0.354 0.314]  (배경 기여 70%)

# %% [markdown]
# ## 4. clamp가 필요한 이유
#
# SH 평가로 나온 색은 이론상 [0,1]을 벗어날 수 있어서(음수·1 초과),
# 합성 후 `torch.clamp(rgb, 0.0, 1.0)`으로 잘라 유효한 이미지로 만든다.

# %%
hot = np.array([1.3, -0.1, 0.5])  # SH 평가가 만들 수 있는 범위 밖 색 (premultiplied 가정)
out = np.clip(hot + (1 - 0.9) * np.ones(3), 0, 1)
print("clamp 전:", (hot + 0.1 * np.ones(3)).round(3), "-> clamp 후:", out.round(3))
# 출력: clamp 전: [ 1.4  0.   0.6] -> clamp 후: [1.  0.  0.6]

# %% [markdown]
# ## 5. 시각화
#
# 왼쪽부터: premultiplied `render`, `alpha` 맵, 그리고 배경 3종으로 합성한 최종 `rgb`.
# 가우시안이 없는 영역이 배경색으로 채워지는 것이 핵심이다.

# %%
fig = make_subplots(
    rows=1,
    cols=5,
    subplot_titles=[
        "render (premultiplied)",
        "alpha (accumulation)",
        "rgb | bg=black",
        "rgb | bg=white",
        f"rgb | bg=random {backgrounds['random'].round(2)}",
    ],
    horizontal_spacing=0.02,
)


def to_img(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


fig.add_trace(go.Image(z=to_img(render)), row=1, col=1)
fig.add_trace(go.Image(z=to_img(np.repeat(alpha, 3, axis=-1))), row=1, col=2)
fig.add_trace(go.Image(z=to_img(composited["black"])), row=1, col=3)
fig.add_trace(go.Image(z=to_img(composited["white"])), row=1, col=4)
fig.add_trace(go.Image(z=to_img(composited["random"])), row=1, col=5)

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    title_text="rgb = render + (1 - alpha) * background  →  clamp[0,1]",
    width=1400,
    height=340,
    margin=dict(l=10, r=10, t=80, b=10),
    font=dict(size=12),
)
_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("saved:", png_path)
# 출력: saved: .../fm-2/splatfacto/.fm/hints/c0240027-1c75-4fbb-9ea7-bcc388b152b3/expy.png

# %% [markdown]
# ## 정리
#
# - `rasterization()`의 `render`는 배경이 빠진 premultiplied 색, `alpha`는 누적 불투명도.
# - 최종 RGB는 $\mathrm{rgb} = \mathrm{render} + (1-\alpha)\cdot\mathrm{bg}$ 로 알파 합성 후 [0,1] clamp.
# - $\alpha$가 낮은(가우시안이 못 덮은) 영역은 배경색으로 채워지고,
#   학습 시 랜덤 배경은 모델이 배경색에 과적합하는 것을 막는다.
