# 컴퓨트 셰이더에서 `groupshared` 변수의 역할

## 한 줄 답

`groupshared`는 **같은 스레드 그룹(16×16 = 256 스레드) 안의 스레드들이 공유하는 메모리**(GPU의 LDS/shared memory)를 선언하는 HLSL 키워드다. 원문의 SH 계수 계산 셰이더에서는 두 가지 용도로 쓰인다.

```hlsl
groupshared float3 SharedCoeffs[ThreadGroupX * ThreadGroupY][9]; // 256 x 9
groupshared int    TotalSample;
```

| 변수 | 용도 |
|---|---|
| `SharedCoeffs[256][9]` | 각 스레드가 자기 담당 샘플로 부분 합산한 **9개 SH 계수(RGB float3)** 를 자기 슬롯(`sharedIndex`)에 내려놓는 자리 |
| `TotalSample` | 모든 스레드가 처리한 **샘플 개수의 총합**. `InterlockedAdd`로 원자적으로 누적 |

마지막에 `GI == 0`(그룹 인덱스 0번) 스레드 한 개가 256개 슬롯을 모두 더하고, `TotalSample`로 `dOmega`를 만들어 최종 `Coeffs[i]`를 기록한다.

## 왜 이런 구조가 필요한가

이 셰이더가 계산하려는 것은 큐브맵 전체에 대한 적분이다.

$$L_{lm} = \frac{2\pi}{n_1 n_2}\sum_{\phi}\sum_{\theta} L(\theta,\phi)\, y_l^m(\theta,\phi)\,\sin\theta$$

구 전체를 `SampleDelta = 0.025` 간격으로 훑으면 샘플 수가 매우 많다. 이를 한 스레드가 순차적으로 돌면 느리므로 **256개 스레드가 (φ, θ) 격자를 나눠서 각자 부분 합을 계산**한다.

- 스레드 `(GTid.x, GTid.y)`는 `phi = GTid.x * SampleDelta`에서 시작해 `DeltaPhi = 0.025 * 16`씩, `theta = GTid.y * SampleDelta`에서 시작해 `DeltaTheta = 0.025 * 16`씩 건너뛴다. 즉 16×16 격자를 interleave 방식으로 분할한다.
- 각 스레드는 자기 샘플들에 대해 `coeffs[i] += radiance * y[i] * sin(theta)`를 레지스터(로컬 변수)에 누적한다.

문제는 **부분 합들을 하나로 합쳐야 한다**는 것이다. 스레드의 로컬 변수는 다른 스레드가 볼 수 없다. 이때 스레드 그룹 내에서 데이터를 교환할 수 있는 유일한 빠른 수단이 `groupshared` 메모리다. 이것이 전형적인 **병렬 리덕션(parallel reduction)** 패턴이다.

## 코드 흐름 단계별

```hlsl
[numthreads(16, 16, 1)]
void main( uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    // (1) 초기화: 한 스레드만 공유 카운터를 0으로
    if ( GI == 0 ) TotalSample = 0;
    GroupMemoryBarrierWithGroupSync();   // 초기화가 끝나길 모두 대기

    // (2) 각 스레드가 자기 샘플들로 부분 합 (레지스터에)
    float3 coeffs[9] = { Black, ... };
    int numSample = 0;
    for ( phi ... ) for ( theta ... ) {
        ... coeffs[i] += radiance * y[i] * sin(theta);
        ++numSample;
    }

    // (3) 부분 합을 공유 메모리의 내 슬롯에 기록
    int sharedIndex = GTid.y * ThreadGroupX + GTid.x;   // 0..255
    for ( i < 9 ) { SharedCoeffs[sharedIndex][i] = coeffs[i]; coeffs[i] = Black; }

    // (4) 샘플 수는 원자적으로 누적
    InterlockedAdd( TotalSample, numSample );
    GroupMemoryBarrierWithGroupSync();   // 256개 슬롯이 모두 채워지길 대기

    // (5) 스레드 0번이 최종 합산 및 출력
    if ( GI == 0 ) {
        for ( i < 256 ) for ( j < 9 ) coeffs[j] += SharedCoeffs[i][j];
        float dOmega = 2.f * PI / float( TotalSample );
        for ( i < 9 ) Coeffs[i] = coeffs[i] * dOmega;
    }
}
```

### 포인트별 설명

1. **`GI == 0`만 `TotalSample = 0`** — 256개 스레드가 동시에 0을 써도 결과는 같지만, 관례상 대표 스레드 하나가 초기화한다. 중요한 건 그 뒤의 `GroupMemoryBarrierWithGroupSync()`: 다른 스레드가 `InterlockedAdd`를 하기 전에 0 초기화가 반드시 보이도록 보장한다.
2. **`sharedIndex = GTid.y * 16 + GTid.x`** — 2D 스레드 ID를 1D 슬롯 번호로 바꾼다. 사실 이 값은 `GI`(`SV_GroupIndex`)와 동일하다. 각 스레드가 **서로 다른 슬롯**에 쓰므로 계수 배열 쪽에는 원자 연산이 필요 없다.
3. **`InterlockedAdd(TotalSample, numSample)`** — 반대로 `TotalSample`은 256개 스레드가 **같은 주소**에 더하므로 원자 연산이 필수다. 각 스레드의 루프 반복 횟수가 부동소수점 경계 때문에 미세하게 다를 수 있어서, 하드코딩 대신 실제 샘플 수를 세어 모은다.
4. **두 번째 배리어** — 모든 스레드가 (3)(4)를 마쳤음을 보장한 뒤에야 0번 스레드가 읽기를 시작할 수 있다. 배리어가 없으면 아직 안 채워진 슬롯을 읽는 레이스가 발생한다.
5. **`dOmega = 2π / TotalSample`** — 수식의 $\frac{2\pi}{n_1 n_2}$ 항에 해당한다. `TotalSample = n1 * n2`이므로 샘플 하나가 차지하는 (1/π 가중이 포함된) 입체각 비율이다. 최종 `Coeffs[i]`는 RGB 각각 9개, 총 27개 계수가 된다.

## `groupshared`의 특성 정리

- **범위**: 한 스레드 그룹(`[numthreads]`로 정의된 집합) 내에서만 공유. 다른 그룹과는 공유되지 않는다. 이 셰이더는 그룹 1개만 디스패치하는 구조라 문제 없다.
- **속도**: 온칩 메모리라 전역 `RWStructuredBuffer`(VRAM)보다 훨씬 빠르다. 그래서 중간 결과 교환에 쓴다.
- **크기 제한**: D3D11/SM5 기준 그룹당 최대 32KB. 여기서는 `256 * 9 * sizeof(float3) = 256 * 9 * 12 = 27,648B` + `int 4B`로 제한 안에 들어간다.
- **동기화 필수**: 쓴 값을 다른 스레드가 읽으려면 `GroupMemoryBarrierWithGroupSync()`가 필요하다. 같은 주소에 여러 스레드가 쓰면 `Interlocked*` 원자 함수를 써야 한다.
- **수명**: 디스패치가 끝나면 사라진다. 결과를 남기려면 반드시 UAV(`Coeffs`)에 써야 한다.

## 요약 암기 포인트

- `SharedCoeffs[256][9]` = 스레드별 부분 SH 계수(9개) 보관소 → **각자 다른 슬롯**, 원자 연산 불필요
- `TotalSample` = 총 샘플 수 → **같은 주소**, `InterlockedAdd` 필요
- 배리어 두 번(초기화 후 / 기록 후) → 마지막에 `GI == 0`이 256개 합산 후 `× 2π/TotalSample`로 `Coeffs` 출력
