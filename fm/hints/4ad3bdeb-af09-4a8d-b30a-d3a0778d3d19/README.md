# step 0에서 `features_rest`와 `quats`의 그래디언트가 거의 0인 이유

> **Q.** step 0에서 `features_rest`와 `quats`의 그래디언트가 거의 0인 이유는?
>
> **A.** `features_rest`는 `sh_degree_to_use=0`이라 고차 밴드가 forward에 쓰이지 않아 grad=0이다. `quats`는 초기 가우시안이 구(isotropic)라서 회전이 렌더에 영향을 주지 않아 grad≈1e-13이다.

두 파라미터 모두 "학습되지 않는" 것처럼 보이지만 원인은 완전히 다릅니다. 하나는 **계산 그래프에 아예 들어가지 않은 것**(정확히 0), 다른 하나는 **그래프에는 들어갔지만 미분값이 수학적으로 0인 것**(부동소수점 오차 수준의 잔여값)입니다.

`splatfacto_train_step.py` F단계에서 `loss.backward()` 직후 출력되는 표를 보면 아래와 같은 패턴이 관찰됩니다.

| group | |grad| mean | 원인 |
|---|---|---|
| `features_rest` | **정확히 0** | forward에 미사용 (그래프 밖) |
| `quats` | **≈1e-13** | Σ = s²I 가 R에 불변 → 해석적으로 0, float 오차만 남음 |

---

## 1. `features_rest`: 사용하지 않은 SH 밴드는 그래프 밖에 있다

### SH 차수 스케줄

splatfacto는 spherical harmonics(SH) 차수를 점진적으로 올립니다 (`nerfstudio/models/splatfacto.py`):

```python
sh_degree_interval: int = 1000          # config 기본값
...
sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
```

- step 0 ~ 999 → degree 0 (계수 1개, `features_dc`만)
- step 1000 ~ 1999 → degree 1 (계수 4개)
- step 2000 ~ 2999 → degree 2 (계수 9개)
- step 3000 이후 → degree 3 (계수 16개, 전부)

`features_dc`는 0차 밴드(`[N,1,3]`), `features_rest`는 1~3차 밴드(`[N,15,3]`)를 담습니다. 분석 스크립트 D4에서는 이 둘을 붙여 `[N,16,3]` 텐서를 만들지만, `gsplat.rasterization(..., sh_degree=sh_degree_to_use)` 에 넘기면 래스터라이저는 **앞 `(sh_degree_to_use+1)²` 개 밴드만 읽습니다**. step 0에서는 1개, 즉 `features_dc` 슬라이스만 씁니다.

### 왜 "정확히" 0인가

autograd는 forward에서 실제로 연산에 참여한 텐서에 대해서만 역전파 경로를 만듭니다. `torch.cat`으로 붙인 텐서의 뒤쪽 15개 밴드는 SH 평가 커널 안에서 곱해지거나 더해지는 일이 전혀 없으므로, 출력 색상에 대한 편미분이 구조적으로 0입니다:

$$
c = \sum_{\ell=0}^{L} \sum_{m} k_{\ell m}\, Y_{\ell m}(d), \qquad L = \texttt{sh\_degree\_to\_use} = 0
\quad\Rightarrow\quad
\frac{\partial c}{\partial k_{\ell m}} = 0 \;\; (\ell \ge 1)
$$

`cat`의 backward는 상류 grad를 각 입력 슬라이스로 잘라 돌려주는데, 뒤쪽 슬라이스로 흘러오는 값이 존재하지 않으니 `features_rest.grad`는 **연산 결과가 0이 아니라 아예 기여가 없는 0**입니다. 그래서 1e-13 같은 노이즈조차 없이 정확히 0.0이 찍힙니다.

### 언제부터 학습되나

step 1000에 `sh_degree_to_use`가 1이 되는 순간부터 1차 밴드 3개가 forward에 들어가고, 그때부터 `features_rest`의 해당 행에 grad가 흐릅니다. 2·3차 밴드는 각각 step 2000, 3000부터입니다. 이 "저차 → 고차" 순서는 원 3DGS 논문의 warm-up 전략을 그대로 따른 것으로, 먼저 base color를 잡고 나중에 view-dependent 성분을 얹어 초기 오버피팅을 막습니다.

> 참고: Adam 관점에서 grad가 정확히 0이면 `exp_avg`, `exp_avg_sq`도 0으로 유지되고 갱신량 $\hat m/(\sqrt{\hat v}+\epsilon) = 0/\epsilon = 0$ 이라 파라미터가 한 치도 움직이지 않습니다. G단계 표에서 `features_rest`의 `|Δparam|`이 0인 이유입니다.

---

## 2. `quats`: 구는 돌려도 구다

### 초기화가 왜 구(isotropic)인가

`populate_modules`에서 초기 scale은 **3-최근접 이웃 거리의 평균 하나를 3축에 똑같이 복사**해 만듭니다:

```python
distances, _ = k_nearest_sklearn(means.data, 3)
avg_dist = distances.mean(dim=-1, keepdim=True)          # [N,1]
scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))   # [N,3], 세 축 동일
quats = torch.nn.Parameter(random_quat_tensor(num_points))      # 무작위 회전
```

즉 모든 가우시안이 $s_x = s_y = s_z = s$ 인 완전한 구로 시작합니다. 쿼터니언은 무작위지만, 아래에서 보듯 구에서는 그 값이 무엇이든 상관없습니다.

### 공분산이 회전에 불변임을 보이기

3DGS의 공분산은 스케일 행렬과 회전 행렬로 파라미터화됩니다:

$$
\Sigma = R\, S S^{\top} R^{\top}, \qquad S = \mathrm{diag}(s_x, s_y, s_z), \quad R = R(q)
$$

isotropic이면 $S S^\top = s^2 I$ 이므로

$$
\Sigma = R\,(s^2 I)\,R^{\top} = s^2\, R R^{\top} = s^2 I
$$

회전 행렬은 직교($R R^\top = I$)이므로 $R$이 식에서 완전히 사라집니다. $\Sigma$가 $q$에 의존하지 않으니

$$
\frac{\partial \Sigma}{\partial q} = 0
\;\Rightarrow\;
\frac{\partial \mathcal L}{\partial q}
= \frac{\partial \mathcal L}{\partial \Sigma}\,\frac{\partial \Sigma}{\partial q} = 0
$$

카메라로 투영한 2D 공분산 $\Sigma' = J W \Sigma W^\top J^\top$ 도 $\Sigma$를 통해서만 $q$를 보기 때문에 마찬가지로 0입니다.

### 그런데 왜 정확히 0이 아니고 1e-13인가

`features_rest`와 달리 `quats`는 forward에 **실제로 참여**합니다. gsplat 커널은 $q$를 정규화 → $R(q)$ 구성 → $R S S^\top R^\top$ 곱셈을 실제로 수행하고, backward도 그 경로를 따라 $\partial \mathcal L / \partial q$를 계산합니다. 해석적으로는 $R S S^\top R^\top$ 의 backward에서 $R$ 방향 성분들이 서로 정확히 상쇄되어야 하지만, 부동소수점 연산에서는

- $s^2$가 float32로 표현되는 과정의 반올림
- $R R^\top$ 가 정확히 $I$가 아니라 $I + O(\epsilon)$ 인 점
- 곱셈 순서에 따른 누적 오차

때문에 상쇄가 완벽하지 않아 $10^{-13}$ 수준의 잔여값이 남습니다. 이것은 "약한 신호"가 아니라 **수치 노이즈**이며, 학습 신호로서 의미가 없습니다.

> Adam은 첫 스텝에 $\hat m_1/\sqrt{\hat v_1} = g/|g| = \pm 1$ 로 정규화하기 때문에, grad가 1e-13이라도 방향만 있으면 정확히 lr 만큼 움직입니다. 그래서 G단계 표에서 `quats`의 `|Δparam|`은 0이 아니라 lr(≈1e-3) 근처로 찍힙니다 — 하지만 그 방향은 float 노이즈의 부호라 의미 없는 랜덤 워크입니다. 구인 동안은 어떻게 돌려도 렌더가 같으니 손해도 없습니다.

### 언제부터 의미가 생기나

`scales`는 step 0부터 정상적인 grad를 받고, 3축이 **각각 독립적으로** 갱신됩니다. 몇 스텝만 지나면 $s_x \ne s_y \ne s_z$ 가 되어 가우시안이 타원체로 변하고, 그 순간부터

$$
\Sigma = R\,\mathrm{diag}(s_x^2, s_y^2, s_z^2)\,R^\top
$$

가 $R$에 의존하게 되어 `quats`에 실질적인 grad가 흐르기 시작합니다. 즉 `quats`의 학습은 "스케줄"이 아니라 **`scales`의 비등방성이 생기는 것에 의해 자연스럽게 켜집니다**. 반대로 말하면, 초기 무작위 쿼터니언은 어떤 값이든 결과에 영향을 주지 않으므로 `random_quat_tensor`로 아무렇게나 뽑아도 문제가 없습니다.

---

## 3. 두 현상 비교 정리

| | `features_rest` | `quats` |
|---|---|---|
| grad 값 | 정확히 0.0 | ≈1e-13 (float 노이즈) |
| 0인 이유 | forward에서 읽히지 않음 → 그래프 밖 | forward에 참여하지만 $\Sigma = s^2 I$ 라 $\partial\Sigma/\partial q = 0$ |
| 노이즈가 없는/있는 이유 | 연산 자체가 없어서 상쇄할 것도 없음 | 실제 곱셈·역전파를 거치며 반올림 오차 잔여 |
| 학습 시작 조건 | step 1000/2000/3000 (`sh_degree_interval` 스케줄, 결정적) | `scales`가 비등방성이 되는 시점 (데이터 의존, 수 스텝 내) |
| step 0 Adam Δparam | 0 | ≈lr (노이즈 부호대로 이동, 렌더에 영향 없음) |

핵심 구별점: **"계산에 없어서 0"** 과 **"계산에 있지만 미분이 0"** 은 출력 숫자로 구분할 수 있습니다. 전자는 완벽한 0, 후자는 기계 정밀도 수준의 잔여값입니다.

---

## 참고 소스

- `.fm/assets/splatfacto_train_step.py` — D4(SH 차수 결정), F(backward 후 grad 표와 관찰 문구), G(Adam step 후 `notes` dict)
- `nerfstudio/models/splatfacto.py` — `populate_modules` (isotropic 초기 scale, `random_quat_tensor`), `sh_degree_interval` config, `get_outputs` 내 `sh_degree_to_use` 계산
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023) — $\Sigma = RSS^\top R^\top$ 파라미터화와 SH warm-up 전략
