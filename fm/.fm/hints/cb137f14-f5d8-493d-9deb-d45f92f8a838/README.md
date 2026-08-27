# Trainer가 모델에 optimizer를 콜백으로 주입해야 하는 이유

## 질문
Trainer가 모델에 optimizer를 콜백으로 주입해야 하는 이유는?

## 답
I단계의 densification이 가우시안 개수를 바꾸면 Adam의 모멘트 텐서(`exp_avg`, `exp_avg_sq`)도 같은 크기로 늘려야 하기 때문이다. 파라미터 **값**은 G(optimizer step)에서만, 파라미터 **개수**는 I(AFTER 콜백)에서만 바뀌므로, I가 optimizer 상태에 접근할 수 있어야 한다.

---

## 1. 배경: 한 스텝 A~I 에서 optimizer가 등장하는 자리

`Trainer.train()`은 매 스텝 다음 순서로 진행한다 (`nerfstudio/engine/trainer.py:259-272`).

| 단계 | 호출 | optimizer와의 관계 |
|---|---|---|
| **A** | `model.step_cb(optimizers, step)` (BEFORE 콜백) | `model.optimizers`, `model.schedulers`에 Trainer의 객체를 **그대로 참조**로 꽂아 준다 |
| B~F | zero_grad → data → forward → loss → backward | grad만 채움 |
| **G** | `optimizers.optimizer_scaler_step_some(...)` | Adam이 파라미터 **값**을 갱신, `exp_avg`/`exp_avg_sq` 갱신 |
| H | `scheduler_step_all(step)` | lr 감쇠 |
| **I** | `model.step_post_backward(step)` (AFTER 콜백) → `strategy.step_post_backward(params, optimizers, state, ...)` | 가우시안 **개수**를 바꾸면서 optimizer 상태도 함께 손댐 |

핵심 관찰은 마지막 행이다. 일반적인 학습 루프에서는 모델이 optimizer를 알 필요가 전혀 없다. 모델은 파라미터와 forward만 제공하고, optimizer는 Trainer가 소유한다. splatfacto가 이 관례를 깨는 유일한 이유가 **I단계의 densification** 이다.

## 2. 왜 파라미터 개수가 바뀌면 optimizer 상태가 문제인가

Adam은 파라미터 `p` 하나마다 `optimizer.state[p]` 딕셔너리에 다음을 유지한다.

- `step` — 스칼라 카운터
- `exp_avg` ($m_t$, 1차 모멘트) — **`p`와 같은 shape**
- `exp_avg_sq` ($v_t$, 2차 모멘트) — **`p`와 같은 shape**

갱신식 $\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$ 는 원소 단위(elementwise) 연산이라, `means`가 `[N,3]`이면 `exp_avg`도 정확히 `[N,3]`이어야 한다.

I단계에서 gsplat `DefaultStrategy`가 하는 일:

- **grow (100스텝마다, step > 500):** 평균 화면공간 그래디언트 $\bar g_i = \text{grad2d}_i/\text{count}_i > 0.0008$ 인 가우시안을 크기에 따라 **복제(clone)** 하거나 **분할(split)** → N 증가
- **prune:** $\sigma(\tilde\alpha_i) < 0.1$ 인 것, (step < 4000) 화면공간 반경이 너무 큰 것 삭제 → N 감소

즉 `gauss_params["means"]`가 `[N,3]`에서 `[N',3]`으로 바뀐다. 이때 optimizer 안의 `exp_avg`가 여전히 `[N,3]`이면 다음 G단계에서 shape 불일치로 바로 깨진다. 또 새 `Parameter` 객체를 만들었으므로 `optimizer.param_groups[i]["params"]`가 가리키는 **옛 텐서 객체**도 교체해야 한다 — 그렇지 않으면 Adam은 더 이상 어떤 forward에도 쓰이지 않는 옛 파라미터를 갱신하게 된다.

정리하면 개수를 바꾸는 쪽(I)이 반드시 다음 세 가지를 동시에 해야 한다.

1. 새 `Parameter` 생성 (`params[name] = p_new`)
2. `optimizer.param_groups[i]["params"] = [p_new]` 로 참조 교체
3. `optimizer.state[p_new]`에 **같은 인덱스로 재배열/확장된** `exp_avg`, `exp_avg_sq` 넣기

2, 3은 optimizer 객체 없이는 불가능하다. 그래서 A에서 미리 꽂아 두어야 한다.

## 3. 코드로 확인

### A단계 — 주입 (`nerfstudio/models/splatfacto.py:407-410`)

```python
def step_cb(self, optimizers: Optimizers, step):
    self.step = step
    self.optimizers = optimizers.optimizers   # dict[str, torch.optim.Optimizer], 복사 아님
    self.schedulers = optimizers.schedulers
```

노트북 A셀에서 `model.optimizers is opt.optimizers → True` 로 확인한다. 같은 객체를 공유하므로 모델이 내부를 바꾸면 Trainer가 다음 G단계에서 쓰는 optimizer가 그대로 바뀐 상태다.

### I단계 — 사용 (`splatfacto.py:365-375`)

```python
def step_post_backward(self, step):
    assert step == self.step
    self.strategy.step_post_backward(
        params=self.gauss_params,
        optimizers=self.optimizers,     # ← A에서 받아둔 것
        state=self.strategy_state, step=self.step, info=self.info, packed=False,
    )
```

### gsplat `_update_param_with_optimizer` — 실제로 리사이즈하는 곳

```python
for name in names:
    optimizer = optimizers[name]
    for i, param_group in enumerate(optimizer.param_groups):
        p = param_group["params"][0]
        p_state = optimizer.state[p]
        del optimizer.state[p]
        for key in p_state.keys():
            if key != "step":                       # exp_avg, exp_avg_sq 만
                p_state[key] = optimizer_fn(key, p_state[key])
        p_new = param_fn(name, p)
        optimizer.param_groups[i]["params"] = [p_new]
        optimizer.state[p_new] = p_state
        params[name] = p_new
```

`param_fn`과 `optimizer_fn`은 파라미터와 모멘트에 **똑같은 인덱스 연산**을 적용한다.

| 연산 | `param_fn` | `optimizer_fn` (exp_avg / exp_avg_sq) |
|---|---|---|
| duplicate | `cat([p, p[sel]])` | `cat([v, zeros(len(sel), ...)])` |
| split | `cat([p[rest], p_split])` | `cat([v[rest], zeros(2*len(sel), ...)])` |
| remove | `p[sel]` | `v[sel]` |

새로 생긴 가우시안의 모멘트는 **0으로 초기화**되고(`zeros`), 살아남은 가우시안의 모멘트는 그대로 따라간다. `step` 카운터는 건드리지 않아 bias-correction은 이어진다.

## 4. 700스텝 실험에서 보이는 것

노트북 마지막 셀은 A~I를 `train_step()` 하나로 묶어 step 1~700을 돌린다. warmup(500) 이후 첫 refine 시점인 step 600과 700에서:

- `가우시안 N → N'` 로 개수가 바뀐다
- `means param shape / Adam exp_avg shape` 가 **함께** `[N,3] → [N',3]` 으로 바뀐다
- `gauss_params['means'] is param_groups[0]['params'][0] → True` — 파라미터 객체가 교체됐지만 optimizer도 새 객체를 가리킨다
- `strategy_state['grad2d']` 가 새 크기로 리셋된다

이 세 가지가 한 스텝 안에서 동시에 맞아 떨어지는 것이 "모델이 optimizer를 알고 있어야 한다"의 실증이다.

## 5. 한 줄 요약

> G는 값을 바꾸고 I는 개수를 바꾼다. 개수를 바꾸면 Adam의 `exp_avg`/`exp_avg_sq`와 `param_groups`의 참조까지 같은 인덱스로 고쳐야 하는데, 그 일을 하는 I가 모델 콜백 안에 있으므로 Trainer는 A(`step_cb`)에서 optimizer를 모델에 넘겨 둔다.

## 참고
- `analysis/splatfacto_train_step.py` — 섹션 A, G, I, 마지막 700스텝 실험
- `nerfstudio/models/splatfacto.py` — `step_cb`, `step_post_backward`, `get_training_callbacks`
- `nerfstudio/engine/trainer.py:259-272` — BEFORE/AFTER 콜백 호출 위치
- `gsplat/strategy/ops.py` — `_update_param_with_optimizer`, `duplicate`, `split`, `remove`
