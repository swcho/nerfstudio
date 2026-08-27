# Adam이 원소별(element-wise)로 독립이라는 성질이 densification에 왜 중요한가?

## 한 줄 답

가우시안 개수 $N$이 바뀌어도 **각 행의 $(m, v)$를 파라미터와 같은 인덱스로 옮기기만 하면** 옵티마이저가 그대로 이어서 동작하기 때문이다. gsplat은 grow/prune 때 `exp_avg`/`exp_avg_sq`를 파라미터와 똑같이 재배열하고, 새로 생긴 가우시안의 모멘트는 0으로 초기화한다.

---

## 1. 왜 문제가 되는가 — 학습 중에 파라미터 shape이 바뀐다

Splatfacto의 파라미터는 `means [N,3]`, `scales [N,3]`, `quats [N,4]`, `opacities [N]`, `features_dc [N,3]`, `features_rest [N,15,3]` 여섯 그룹이고, 그룹마다 **별도의 Adam 인스턴스**가 붙어 있다 (`splatfacto_train_step.py` G단계).

Adam은 첫 `step()` 직후 파라미터와 **같은 shape**의 상태 텐서 `exp_avg`($m$), `exp_avg_sq`($v$)를 만든다. 문제는 I단계(`strategy.step_post_backward`)에서 100스텝마다 다음이 일어난다는 것이다.

| 연산 | 파라미터 변화 | $N$ 변화 |
|---|---|---|
| duplicate (clone) | `cat([p, p[sel]])` | $N \to N + n_{dup}$ |
| split | `cat([p[rest], p_split])` (원본 삭제, 2개 생성) | $N \to N - n_{split} + 2n_{split}$ |
| remove (prune) | `p[keep]` | $N \to N - n_{prune}$ |

파라미터 텐서가 `[N,·]`에서 `[N',·]`로 바뀌면 `[N,·]` 크기의 $m, v$는 그대로 쓸 수 없다. 이때 선택지는 둘이다.

1. 옵티마이저 상태를 통째로 버리고 새로 시작한다 → 100스텝마다 모멘텀이 리셋되어 학습이 불안정해진다.
2. 상태를 파라미터와 **같은 방식으로 재배열**한다 → 이게 가능하려면 상태가 "행 단위로 분리 가능"해야 한다.

## 2. Adam 갱신식이 왜 행 단위로 분리 가능한가

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon}
$$

이 식에서 파라미터의 $i$번째 원소 $\theta_i$의 갱신은 **오직 $m_i, v_i, g_i$만** 참조한다. 다른 원소의 값이나 그래디언트는 전혀 섞이지 않는다 (유일하게 공유되는 것은 스칼라 `step` 카운트 $t$뿐).

따라서 가우시안 $i$에 해당하는 행 $(\theta_i, m_i, v_i)$는 하나의 독립된 묶음이다. 이 묶음을

- 다른 위치로 옮겨도 (permutation),
- 지워도 (prune),
- 새 행을 끼워 넣어도 (grow)

기존 가우시안들의 갱신 궤적은 **정확히 이전과 동일**하게 이어진다. 그저 인덱스만 바뀐 것이다.

## 3. gsplat이 실제로 하는 일 — `strategy/ops.py`

세 연산 모두 같은 헬퍼 `_update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)`를 쓴다. 핵심 부분은 다음과 같다.

```python
for name in names:
    optimizer = optimizers[name]
    for i, param_group in enumerate(optimizer.param_groups):
        p = param_group["params"][0]
        p_state = optimizer.state[p]
        del optimizer.state[p]                       # 옛 Parameter 키 제거
        for key in p_state.keys():
            if key != "step":                        # 스칼라 step은 그대로 둠
                p_state[key] = optimizer_fn(key, p_state[key])   # exp_avg, exp_avg_sq 재배열
        p_new = param_fn(name, p)                    # 새 Parameter 생성
        optimizer.param_groups[i]["params"] = [p_new]
        optimizer.state[p_new] = p_state             # 새 Parameter 키로 상태 등록
        params[name] = p_new
```

`param_fn`과 `optimizer_fn`이 **같은 인덱스 규칙**을 쓰는 것이 포인트다.

| 연산 | `param_fn` | `optimizer_fn` (exp_avg, exp_avg_sq 각각) |
|---|---|---|
| `duplicate` | `cat([p, p[sel]])` | `cat([v, zeros(len(sel), ...)])` |
| `split` | `cat([p[rest], p_split])` | `cat([v[rest], zeros(2*len(sel), ...)])` |
| `remove` | `p[sel]` (sel = 살아남는 것) | `v[sel]` |

- **살아남는 가우시안**: 파라미터가 `p[rest]`/`p[sel]`로 옮겨가면 모멘트도 `v[rest]`/`v[sel]`로 똑같이 옮겨간다. 학습 이력이 보존된다.
- **새 가우시안**: 모멘트를 **0으로 초기화**한다. 복제된 가우시안이 원본의 모멘트를 그대로 물려받는 것도 가능하겠지만, gsplat은 0을 택했다. 0에서 시작하면 첫 스텝에 $\hat m_1 = g_1$, $\hat v_1 = g_1^2$이라 갱신량이 $\pm\eta$가 되어, 새 가우시안은 "lr 크기로 새 출발"한다.
- **`step` 카운트**: 스칼라라서 그대로 유지된다. 즉 새 가우시안도 bias correction은 $t$가 큰 상태로 시작하는데, 모멘트가 0이므로 실질적 영향은 없다.

`DefaultStrategy._grow_gs`는 `duplicate` → `split` 순으로, `_prune_gs`는 `remove`를 호출하며, 함께 `strategy_state["grad2d"]`, `["count"]`, `["radii"]` 같은 부수 상태도 같은 인덱스 규칙으로 재배열한다.

## 4. 왜 A단계에서 옵티마이저를 모델에 꽂아 주는가

`Trainer`는 매 스텝 `model.step_cb(optimizers, step)`으로 옵티마이저를 모델에 주입한다. 이유는 위 코드가 보여 준다. `_update_param_with_optimizer`는 `optimizer.param_groups[i]["params"]`를 **새 Parameter 객체로 교체**하고 `optimizer.state`의 키도 바꿔야 하므로, 모델(strategy)이 옵티마이저 객체에 직접 접근해야 한다. 파라미터만 바꾸고 옵티마이저를 안 건드리면 옵티마이저는 이미 그래프에서 떨어진 옛 텐서를 계속 갱신하게 된다.

## 5. 700스텝 실험이 보여 주는 것

`splatfacto_train_step.py` 마지막 셀은 A~I를 `train_step()` 하나로 묶어 step 1~700을 돈다. warmup(500) 이후 step 600, 700에서 첫 refine이 일어나며 출력되는 항목은

- `가우시안 N → N'` : 개수 변화
- `means param / Adam exp_avg shape: ((N,3),(N,3)) → ((N',3),(N',3))` : 파라미터와 모멘트가 **함께** 커진 것
- `gauss_params['means'] is param_groups[0]['params'][0] → True` : Parameter 객체가 교체됐지만 모델과 옵티마이저가 같은 객체를 가리킴
- `state['grad2d']` shape이 N'으로 바뀌고 sum=0 : 통계 상태도 리셋됨

loss 곡선이 step 600, 700에서 튀지 않는 것이 "기존 가우시안의 모멘트가 보존됐다"는 간접 증거다.

## 6. 대비 — 원소별로 독립이 아니면 무엇이 깨지는가

이 재배열 트릭은 **상태가 파라미터와 같은 shape의 텐서이고, 원소 간 결합이 없는** 옵티마이저(SGD+momentum, RMSprop, Adam, AdamW, Adagrad 등)에서만 성립한다.

| 옵티마이저 | 상태의 형태 | grow/prune 시 |
|---|---|---|
| Adam / SGD momentum | 파라미터와 같은 shape, 원소별 | 행 인덱싱으로 재배열 가능 |
| L-BFGS | 최근 $k$개의 $(s_j, y_j) = (\Delta\theta, \Delta g)$ 벡터 쌍 → 이걸로 역헤시안 근사를 만듦 | 벡터 길이가 $N$에 묶여 있고, 역헤시안 근사가 **모든 원소를 섞는** 내적/외적으로 계산됨. 행을 지우거나 추가하면 곡률 정보가 무효 → 히스토리를 통째로 버려야 함 |
| Shampoo / K-FAC 등 2차 근사 | 파라미터 차원별 preconditioner 행렬 ($N \times N$ 등) | $N$이 바뀌면 행렬 자체를 다시 만들어야 하고 행/열을 동시에 재배열해도 새 행의 곡률은 알 수 없음 |
| Adafactor (factored) | `[N,·]`을 행 통계 `[N]`과 열 통계 `[·]`로 분해 | 행 통계는 재배열 가능하지만 열 통계가 모든 행의 평균이라 $N$이 바뀌면 왜곡됨 |

즉 L-BFGS처럼 "파라미터 전체를 하나의 벡터로 보고 원소 사이의 관계(곡률)를 기억하는" 옵티마이저는 가우시안 개수가 동적으로 변하는 3DGS 학습과 근본적으로 맞지 않는다. Adam이 3DGS의 사실상 표준이 된 데는 (그래디언트 스케일이 파라미터마다 극단적으로 다르다는 이유 외에도) **원소별 독립성 덕분에 densification과 마찰 없이 어울린다**는 실용적 이유가 있다.

## 정리

1. densification은 100스텝마다 파라미터 텐서의 행 수 $N$을 바꾼다.
2. Adam은 원소별로 독립이므로 상태 $(m, v)$도 파라미터와 같은 shape이고 행 단위로 분리된다.
3. 그래서 gsplat은 `param_fn`과 `optimizer_fn`에 같은 인덱스 규칙을 적용해 살아남는 행은 옮기고, 새 행은 0으로 채운다 (`_update_param_with_optimizer`).
4. 이를 위해 Trainer는 매 스텝 옵티마이저를 모델에 주입한다 (A단계).
5. L-BFGS처럼 원소 간 결합 상태를 가진 옵티마이저라면 이 트릭이 불가능해 매 refine마다 이력을 버려야 한다.
