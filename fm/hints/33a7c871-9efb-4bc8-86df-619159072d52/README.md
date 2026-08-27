# grow/prune 시 optimizer에는 어떤 일이 일어나는가?

## 한 줄 답

가우시안 개수 N이 바뀌면 파라미터 텐서(`[N, ...]`)를 **새 `torch.nn.Parameter`로 통째로 다시 만들고**, gsplat의 `_update_param_with_optimizer`가 그와 짝이 되는 Adam 상태(`exp_avg`, `exp_avg_sq`)를 **같은 인덱스 규칙으로 잘라 붙인 뒤** optimizer의 `param_groups`와 `state`를 새 객체로 갈아 끼운다. 그래서

- `model.gauss_params['means'] is opt.optimizers['means'].param_groups[0]['params'][0]` 는 refine 뒤에도 `True`,
- 새로 생긴 가우시안 행의 모멘트는 **0**으로 들어오고,
- `step` 카운터는 그대로 유지된다.

## 왜 문제가 되는가 — Adam 상태는 파라미터 "객체"에 묶여 있다

`torch.optim.Optimizer.state`는 `dict[Parameter → dict]`이고, 키는 파라미터 객체 자체(`id` 기준)이다. Adam이 첫 `step()` 뒤에 만드는 상태는 파라미터와 **같은 shape**이다(A~G 단계 노트북 출력: `step 후 Adam state: {'means': ['step', 'exp_avg', 'exp_avg_sq'], ...}`).

Adam 갱신식은 원소별로 독립이다:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

따라서 가우시안 i의 파라미터 행이 어디로 옮겨지든, 그 행의 `(m, v)`만 같은 위치로 따라가면 수학적으로 아무 문제가 없다. 반대로 아무것도 안 하면

1. `params[name] = torch.nn.Parameter(new_tensor)` 로 ParameterDict의 항목만 바꾸면 optimizer는 여전히 **옛 객체**를 들고 있어 새 파라미터는 학습되지 않고,
2. 옛 객체에 묶인 `exp_avg`가 shape `[N_old, 3]`이라 다음 `step()`에서 `[N_new, 3]` grad와 브로드캐스트 에러가 난다.

이것이 A단계에서 `model.step_cb(optimizers, step)`로 **모델에 optimizer를 꽂아주는 이유**이고, splatfacto가 `self.optimizers = optimizers.optimizers` (이름 → Adam 인스턴스 dict)를 그대로 strategy에 넘기는 이유다.

## 핵심 코드 — `gsplat/strategy/ops.py::_update_param_with_optimizer`

```python
@torch.no_grad()
def _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=None):
    if names is None:
        names = list(params.keys())            # means, scales, quats, opacities, features_dc, features_rest
    for name in names:
        optimizer = optimizers[name]           # nerfstudio: 파라미터 그룹 하나 = Adam 하나
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]       # 옛 Parameter 객체
            p_state = optimizer.state[p]       # {'step', 'exp_avg', 'exp_avg_sq'}
            del optimizer.state[p]             # ① 옛 키 제거
            for key in p_state.keys():
                if key != "step":              # ② step 카운터는 건드리지 않음
                    p_state[key] = optimizer_fn(key, p_state[key])   # 모멘트를 재배열
            p_new = param_fn(name, p)          # ③ 새 Parameter 객체 생성
            optimizer.param_groups[i]["params"] = [p_new]           # ④ param_groups 교체
            optimizer.state[p_new] = p_state   # ⑤ 새 키로 상태 다시 등록
            params[name] = p_new               # ⑥ ParameterDict 항목 교체
```

읽는 순서:

| 단계 | 하는 일 | 의미 |
|---|---|---|
| ① `del optimizer.state[p]` | 옛 파라미터를 키로 한 상태 항목 삭제 | 옛 객체는 이제 어디서도 참조되지 않아 GC 대상 |
| ② `key != "step"` | `exp_avg`, `exp_avg_sq`만 `optimizer_fn` 적용 | `step`은 스칼라(텐서)라 shape 문제가 없고, bias correction $1/(1-\beta^t)$ 은 전체 파라미터 그룹 공통이므로 이어서 세는 것이 맞음. 0으로 되돌리면 기존 가우시안 전체가 다시 "첫 스텝처럼" 크게 움직이는 부작용이 생긴다 |
| ③ `param_fn(name, p)` | 이름별 규칙으로 새 텐서를 만들고 `torch.nn.Parameter`로 감쌈 | `requires_grad=True`인 새 leaf 텐서 |
| ④ `param_groups[i]["params"] = [p_new]` | optimizer가 다음 `step()`에서 볼 파라미터 목록 교체 | lr, betas 등 그룹 하이퍼파라미터는 그대로 |
| ⑤ `optimizer.state[p_new] = p_state` | 재배열된 모멘트를 새 객체에 다시 매핑 | 키만 바뀌었고 dict 값은 같은 객체 |
| ⑥ `params[name] = p_new` | 모델 쪽 ParameterDict도 같은 객체로 갱신 | 아래 "identity 체크" 참고 |

`param_fn`과 `optimizer_fn`은 **같은 인덱스 규칙**을 두 번 쓰는 것이라, 파라미터 행과 모멘트 행이 항상 1:1로 맞는다.

## 세 연산에서 `param_fn` / `optimizer_fn`이 무엇을 하는가

### `duplicate` (복제, 작은 가우시안 + 큰 grad)

```python
sel = torch.where(mask)[0]
param_fn     = lambda name, p: torch.nn.Parameter(torch.cat([p, p[sel]]))
optimizer_fn = lambda key, v: torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])
```

- 파라미터: 기존 N개 뒤에 선택된 것을 그대로 복사해 붙임 → `[N + |sel|, ...]`
- 모멘트: 기존 N개 뒤에 **0** `|sel|`개. 복제된 가우시안은 원본과 파라미터는 같지만 모멘트는 0에서 새로 시작한다(원본의 모멘트를 복사하는 것도 가능하지만 gsplat/원본 3DGS 구현 모두 0으로 둔다).

### `split` (분할, 큰 가우시안 + 큰 grad)

```python
sel, rest = torch.where(mask)[0], torch.where(~mask)[0]
# param_fn: means는 원본 위치 + R(q)(s ⊙ z), scales는 log(s/1.6), 나머지는 그대로 복제 → 2개
p_new = torch.cat([p[rest], p_split])                     # 원본은 삭제, 뒤에 2|sel|개 추가
# optimizer_fn:
return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]))])
```

- 결과 크기: `N - |sel| + 2|sel| = N + |sel|`
- **순서가 바뀐다**: 분할되지 않은 가우시안(`rest`)이 앞으로 압축되고, 새 2개씩이 뒤에 붙는다. `optimizer_fn`도 `v[rest]`로 정확히 같은 순서를 만들기 때문에 기존 가우시안의 모멘트는 옮겨진 자기 행을 따라간다.
- 분할된 자식 2개의 모멘트는 0. 원본의 모멘트는 원본과 함께 버려진다.

### `remove` (prune, opacity < 0.1 또는 화면 반경 과대)

```python
sel = torch.where(~mask)[0]                  # 남길 것
param_fn     = lambda name, p: torch.nn.Parameter(p[sel])
optimizer_fn = lambda key, v: v[sel]
```

- 파라미터와 모멘트를 같은 `sel`로 인덱싱 → 살아남은 가우시안은 자기 `(m, v)`를 그대로 가져간다.

세 함수 모두 마지막에 `state["grad2d"]`, `state["count"]`, `state["radii"]` 같은 strategy 통계 텐서도 같은 규칙(`cat`, `[rest]`+repeat, `[sel]`)으로 재배열한다. (DefaultStrategy는 refine 직후 이 통계를 0으로 리셋한다 — 700스텝 실험 출력의 `state['grad2d'] 리셋 ... sum=0.0`.)

## 왜 identity 체크가 `True`로 유지되는가

nerfstudio 쪽 배선:

- `SplatfactoModel.get_gaussian_param_groups()`가 `{name: [self.gauss_params[name]]}` 를 돌려주고, `Optimizers.__init__`가 그룹 이름마다 `Adam(params=[그 Parameter])` 를 하나씩 만든다(`optimizers.py:108`). 그래서 각 Adam은 `param_groups`가 하나, 그 안 `params`가 원소 하나다 — `_update_param_with_optimizer`가 `param_group["params"][0]` 하나만 다루는 전제와 정확히 맞는다.
- `step_cb`가 `self.optimizers = optimizers.optimizers` (같은 dict 객체)를 모델에 저장하고, `step_post_backward`는 `strategy.step_post_backward(params=self.gauss_params, optimizers=self.optimizers, ...)` 로 **`gauss_params` ParameterDict 자체**를 `params`로 넘긴다.
- 따라서 ⑥ `params[name] = p_new`는 곧 `model.gauss_params[name] = p_new`이고, ④에서 같은 `p_new`를 `param_groups[0]["params"]`에 넣었으므로 두 곳이 같은 객체를 가리킨다. 모델의 `self.means` 프로퍼티도 `self.gauss_params["means"]`를 읽기 때문에 다음 forward부터 새 파라미터가 사용된다.

노트북(`splatfacto_train_step.py`, 700스텝 실험)에서 확인한 것:

```
[step 600] 가우시안 N_before → N_after
   means param / Adam exp_avg shape: ((N_before,3),(N_before,3)) → ((N_after,3),(N_after,3))
   param 객체가 교체됨: gauss_params['means'] is param_groups[0]['params'][0] → True
   state['grad2d'] 리셋: (N_after,), sum=0.0
```

`adam_state_shapes()`가 `o.param_groups[0]["params"][0]`를 키로 `o.state`를 조회해 `exp_avg` shape를 얻는다는 점 자체가, 새 객체가 `state`의 키로 재등록되었음을 보여준다(옛 객체로 조회했다면 `KeyError`/빈 dict였을 것).

## 새 가우시안의 모멘트가 0이라는 것의 효과

- 복제/분할된 가우시안은 `exp_avg = exp_avg_sq = 0`이지만 **`step` 카운터는 공유**하므로 $t$ 가 크다(예: 600). bias correction $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ 은 $t$ 가 크면 거의 1이라 첫 갱신은 대략 $\eta \cdot \frac{(1-\beta_1) g}{\sqrt{(1-\beta_2)} |g|} = \eta \cdot \frac{0.1}{\sqrt{0.001}} \approx 3.16\,\eta$ 정도로, 기존 가우시안(≈η)보다 몇 배 크게 움직인 뒤 몇 스텝 안에 정상 크기로 수렴한다. 이는 3DGS 원본 구현과 같은 동작이고 실무상 문제로 취급되지 않는다.
- 반대로 `step`을 0으로 리셋했다면 모든 가우시안이 $t=1$ 상태가 되어 $|\Delta\theta| = \eta$ 로 튀는 스텝이 생긴다. 그래서 `key != "step"` 분기가 있다.

## 기억할 요약

1. grow/prune = "새 Parameter 만들기" + "Adam 모멘트 같은 인덱스로 재배열" + "optimizer/ParameterDict를 새 객체로 재연결".
2. 재연결 3종: `del optimizer.state[p]` → `param_groups[i]["params"] = [p_new]` → `optimizer.state[p_new] = p_state`, 그리고 `params[name] = p_new`.
3. 새 행 모멘트는 0, 기존 행 모멘트는 따라감, `step`은 유지.
4. `gauss_params`가 strategy에 넘어가는 바로 그 dict이므로 `gauss_params['means'] is param_groups[0]['params'][0]`는 항상 `True`.
