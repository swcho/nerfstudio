# 탄젠트 기저에서 `abs(normal.y) < 0.999` 조건을 두는 이유

## 질문
탄젠트 기저를 만들 때 `abs(normal.y) < 0.999` 조건을 두는 이유는?

## 답
normal이 (0,1,0)에 거의 평행하면 up과 외적이 0에 가까워져 기저가 무너지기 때문이다. 이 경우 대신 (0,0,1)을 up으로 사용해 축퇴를 피한다.

---

## 1. 어디에 나오는 코드인가

원문(구면 조화 함수 / Irradiance Map 글)에서 큐브맵을 컨볼루션해 Irradiance Map을 만드는 픽셀 셰이더의 첫 부분이다. 반구 위의 샘플 방향을 **normal을 z축으로 하는 로컬(탄젠트) 좌표계**에서 만들고, 이를 월드 좌표로 회전시켜 큐브맵을 샘플링한다.

```hlsl
float3 normal = normalize( input.localPosition );

float3 up    = ( abs( normal.y ) < 0.999 ) ? float3( 0.f, 1.f, 0.f ) : float3( 0.f, 0.f, 1.f );
float3 right = normalize( cross( up, normal ) );
up           = normalize( cross( normal, right ) );

float3x3 toWorld = float3x3( right, up, normal );
```

이후 루프에서 `tangentSample = (sinθ cosφ, sinθ sinφ, cosθ)` 를 만들어 `mul(tangentSample, toWorld)` 로 월드 방향을 얻는다. 즉 `right`, `up`, `normal` 세 벡터가 **서로 수직인 단위 벡터(정규직교 기저)** 여야 반구 샘플링이 올바르게 동작한다.

## 2. 왜 임의의 "up" 벡터가 필요한가

normal 하나만 주어지면, normal에 수직인 평면 위에서 `right`를 어느 방향으로 잡을지 정할 수 없다(자유도가 1 남음). 그래서 임시 참조 벡터 `up`을 하나 골라

- `right = normalize(cross(up, normal))` — up과 normal 둘 모두에 수직인 벡터
- `up' = normalize(cross(normal, right))` — 실제로 normal에 수직이 되도록 up을 다시 계산(Gram–Schmidt와 같은 효과)

로 나머지 두 축을 만든다. 이 방식은 `up`이 normal과 **평행하지만 않으면** 언제나 잘 동작한다.

## 3. 무엇이 문제인가 — 외적의 축퇴(degeneracy)

외적의 크기는 `|a × b| = |a||b| sin(angle)` 이다. `up = (0,1,0)`을 고정으로 쓰면:

| normal | angle(up, normal) | `cross(up, normal)` 크기 | 결과 |
|---|---|---|---|
| (1,0,0) | 90° | 1 | 정상 |
| (0, 0.7, 0.7) | 45° | 0.7 | 정상 |
| (0, 0.9999, 0.014) | ≈0.8° | ≈0.014 | 매우 작음 → normalize 시 정밀도 손실 |
| (0, 1, 0) | 0° | **0** | `normalize(0,0,0)` → NaN/쓰레기 값 |

normal이 정확히 ±y 방향이면 외적이 영벡터가 되어 `normalize`가 0으로 나누기를 하고, `right`, `up`이 모두 NaN이 된다. 거의 평행한 경우에도 외적이 아주 작아 float 정밀도(특히 셰이더의 `half`/`float`)에서 방향이 크게 흔들리거나 정규화가 불안정하다. 큐브맵 렌더링에서는 **+Y, −Y 면의 중심 픽셀**이 바로 이 경우에 해당하므로, 처리하지 않으면 Irradiance Map의 위/아래 면 한가운데에 검은 점이나 NaN 얼룩이 생긴다.

## 4. 해결 — 축퇴 근처에서만 다른 참조 벡터로 교체

```hlsl
float3 up = ( abs( normal.y ) < 0.999 ) ? float3( 0, 1, 0 ) : float3( 0, 0, 1 );
```

- `abs(normal.y) < 0.999` : normal과 y축 사이 각이 약 2.6° 이상(`acos(0.999) ≈ 2.56°`) → 평소처럼 (0,1,0) 사용.
- 그렇지 않으면(normal이 ±y에 거의 붙어 있음) → **(0,0,1)** 을 참조 벡터로 사용. 이때 normal ≈ ±y 이므로 z축과는 거의 90°를 이루어 외적이 크기 ≈1로 안정적이다.
- `abs()`를 쓰는 이유: +y 뿐 아니라 **−y(0,−1,0)** 도 똑같이 평행(반평행)하여 외적이 0이 되기 때문.

임계값 0.999는 "정확히 평행할 때만"이 아니라 **거의 평행한 근방 전체**를 잡아내기 위한 여유(margin)다. 1.0으로 비교하면 float 오차 때문에 0.99999 같은 값은 통과해 버려 여전히 매우 작은 외적을 정규화하게 된다.

## 5. 왜 교체 후에도 결과가 올바른가

참조 벡터 `up`은 **처음 방향을 결정하는 힌트**일 뿐이고, 두 번의 외적으로 최종 기저는 항상 normal에 정규직교하게 재구성된다. 따라서 (0,1,0)을 쓰든 (0,0,1)을 쓰든 `right`/`up`이 normal 주위로 회전한 값만 달라진다. 이 셰이더는 φ를 0~2π 전체에 대해 적분(합)하므로 반구 기저가 normal 축을 중심으로 얼마나 돌아가 있는지는 결과(Irradiance)에 영향을 주지 않는다. 즉 조건 분기가 만들어내는 불연속은 **결과 값에는 나타나지 않고** 축퇴만 제거한다.

## 6. 관련 지식 / 대안

- 이 트릭은 그래픽스에서 매우 흔한 관용구다. 예: LearnOpenGL의 IBL Irradiance 예제(`up = vec3(0,1,0)` 고정 — 축퇴 위험 있음), PBRT의 `CoordinateSystem()` (|x|>|y| 여부로 분기), Duff et al. 2017 "Building an Orthonormal Basis, Revisited"(분기 없는 안정적 방법), Frisvad 2012.
- 요약하면: **참조 벡터와 normal이 평행하면 외적이 0이 되어 기저가 축퇴하므로, 평행에 가까울 때는 다른 축을 참조 벡터로 바꿔 준다.** 이것이 `abs(normal.y) < 0.999`의 전부다.
