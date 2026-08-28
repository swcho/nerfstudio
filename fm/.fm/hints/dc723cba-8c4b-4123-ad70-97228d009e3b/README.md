# Irradiance 픽셀 셰이더의 탄젠트 공간 기저 구성

## 왜 기저가 필요한가

Irradiance Map 픽셀 셰이더는 큐브맵의 각 텍셀(방향 `normal`)에 대해, 그 방향을 중심으로 한 **반구(hemisphere)** 전체의 큐브맵 값을 `cos(θ)` 가중 적분한다. 반구 샘플은 계산이 편하도록 **"z축이 위"인 탄젠트 공간**에서 구면 좌표로 만든다.

```hlsl
float3 tangentSample = float3( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) );
```

`theta ∈ [0, π/2)`, `phi ∈ [0, 2π)` 이므로 이 벡터는 항상 `z ≥ 0`인 반구 위에 놓인다. 문제는 이 반구를 **실제 텍셀의 `normal` 방향으로 돌려 놓아야** 한다는 것. 그 회전을 담는 것이 `float3x3 toWorld` 이다. 즉 탄젠트 공간의 `(x, y, z)` 축을 월드의 `(right, up, normal)` 축으로 보내는 정규 직교 기저를 만들어야 한다.

## 코드

```hlsl
float3 normal = normalize( input.localPosition );

float3 up = ( abs( normal.y ) < 0.999 ) ? float3( 0.f, 1.f, 0.f ) : float3( 0.f, 0.f, 1.f );
float3 right = normalize( cross( up, normal ) );
up = normalize( cross( normal, right ) );

float3x3 toWorld = float3x3( right, up, normal );
...
float3 worldSample = normalize( mul( tangentSample, toWorld ) );
```

## 단계별 설명

1. **`normal` 결정** — 큐브맵 면의 로컬 위치(`input.localPosition`, 지오메트리 셰이더가 넘겨준 -1~1 범위의 큐브 표면 좌표)를 정규화하면 그 텍셀이 대표하는 방향이 된다. 이 벡터가 기저의 세 번째 축(탄젠트 공간의 z)이 된다.

2. **임시 `up` 선택** — 기저를 만들려면 `normal`과 평행하지 않은 아무 벡터 하나가 필요하다. 보통 월드 Y축 `(0,1,0)`을 쓰는데, `normal`이 거의 ±Y 방향이면(`abs(normal.y) ≥ 0.999`) 두 벡터가 평행해져 외적이 0벡터가 되고 `normalize`가 NaN을 만든다. 그래서 그 경우엔 Z축 `(0,0,1)`로 갈아탄다. 이것이 **특이점(degenerate case) 회피**이고, 큐브맵의 +Y/−Y 면 중앙 텍셀에서 실제로 발생하는 상황이다.

3. **`right = normalize(cross(up, normal))`** — `up`과 `normal` 모두에 수직인 벡터. 임시 `up`은 `normal`과 정확히 직교하지 않지만, 외적 결과는 두 입력 모두에 항상 수직이므로 `right ⟂ normal`이 보장된다. 임시 `up`이 `normal`과 90°가 아니면 외적 길이가 1이 아니므로 `normalize`가 필요하다.

4. **`up = normalize(cross(normal, right))`** — 이제 서로 수직인 `normal`, `right`로부터 진짜 `up`을 다시 계산한다. 이렇게 하면 `up`도 두 축에 정확히 수직이 되어 **세 벡터가 서로 직교하고 길이가 1인 정규 직교(orthonormal) 기저**가 완성된다. 처음의 `(0,1,0)`은 단지 "대략 위쪽"이라는 힌트로만 쓰이고 버려진다. (`normal`, `right`가 이미 단위·직교이면 외적 길이는 1이지만 부동소수 오차를 없애기 위해 한 번 더 `normalize`한다.)

5. **`toWorld = float3x3(right, up, normal)`** — HLSL `float3x3(a, b, c)`는 세 벡터를 **행(row)** 으로 쌓는다. 그리고 `mul(vector, matrix)`는 벡터를 행 벡터로 취급해 `v · M`을 계산하므로, 결과는
   `x*right + y*up + z*normal` 이 된다. 즉 탄젠트 공간 좌표 `(x, y, z)`가 그대로 `right/up/normal` 축의 계수로 해석되어 월드 방향이 나온다. `z = cos(theta) ≥ 0`이므로 샘플은 항상 `normal` 쪽 반구에 놓인다.
   - 외적 순서 `cross(up, normal)` → `cross(normal, right)`는 `(right, up, normal)`이 `(x, y, z)`처럼 **오른손 좌표계**를 이루도록 맞춘 것이다. `x × y = z`에 대응해 `right × up = normal`, `up × normal = right`, `normal × right = up`이 성립한다.

## 자주 헷갈리는 점

- **`normalize`가 두 번 필요한 이유**: 첫 외적은 `up`이 `normal`과 직교하지 않아 길이가 1이 아니고, 두 번째는 오차 정리용이다.
- **왜 `up`을 두 번 계산하나**: 첫 `up`은 "아무 비평행 벡터", 두 번째 `up`이 실제 기저 축이다. 이 "Gram–Schmidt 두 번 외적" 방식은 `learnopengl` Diffuse irradiance 튜토리얼 GLSL 코드와 동일한 패턴이며, 이 글은 그 코드를 HLSL로 옮긴 것이다.
- **회전은 `phi` 방향으로 임의적**: `up` 기준 벡터 선택에 따라 기저가 `normal` 축 둘레로 돌아갈 수 있지만, 반구 전체를 `phi`로 한 바퀴 적분하므로 결과 irradiance는 달라지지 않는다.
- **`0.999` 임계값**: 평행에 가까울 때 외적 길이가 `sin(각도)`로 매우 작아져 정밀도가 떨어지므로 완전히 1이 되기 전에 미리 대체 벡터로 바꾼다.

## 결과

![Irradiance Map(32x32 큐브맵)](fig-1.png)

위 그림은 이 셰이더가 만든 32×32 Irradiance Map 큐브맵을 펼친 것이다. 각 텍셀은 자기 `normal` 방향 반구의 코사인 가중 평균이므로 원본 스카이박스의 디테일은 사라지고, 위(+Y 면, 그림의 상단)가 어둡고 아래(−Y 면, 하단)가 밝은 완만한 그라데이션만 남는다. 텍셀마다 `normal`이 달라도 위의 기저 구성 덕분에 모든 텍셀이 같은 탄젠트 공간 샘플 패턴을 재사용해 자기 반구를 적분할 수 있다.
