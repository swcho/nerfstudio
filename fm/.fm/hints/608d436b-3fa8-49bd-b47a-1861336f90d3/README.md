# 컴퓨트 셰이더에서 `InterlockedAdd`와 `GroupMemoryBarrierWithGroupSync`의 역할

## 문제의 맥락: SH 계수 $L_{lm}$을 병렬로 적분하기

원문의 컴퓨트 셰이더는 큐브맵(환경맵)에서 2차(l ≤ 2) 구면 조화 계수 9개를 구한다.

$$ L_{lm} = \frac{2\pi}{n_1 n_2} \sum_{\phi}\sum_{\theta} L(\theta,\phi)\, y_l^m(\theta,\phi)\, \sin\theta $$

이 이중합을 **16×16 = 256개의 스레드**가 나눠 계산한다. 각 스레드는 `GTid.x`, `GTid.y`로 시작 위치를 어긋나게 잡고 `DeltaPhi = SampleDelta * 16`, `DeltaTheta = SampleDelta * 16` 간격으로 구면을 stride 샘플링한다. 즉 256개의 스레드가 서로 겹치지 않는 샘플 부분집합을 담당하고, 마지막에 **하나로 합쳐야** 한다. 이 "합치기(reduction)" 단계에서 두 동기화 도구가 등장한다.

```hlsl
groupshared float3 SharedCoeffs[ThreadGroupX * ThreadGroupY][9];  // 스레드별 부분합 (256 × 9)
groupshared int    TotalSample;                                    // 그룹 전체 샘플 개수

[numthreads(16, 16, 1)]
void main( uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    if ( GI == 0 ) TotalSample = 0;          // (1) 한 스레드만 초기화
    GroupMemoryBarrierWithGroupSync();       // (2) 초기화가 끝난 뒤에 진행

    // ... 각 스레드가 자기 몫의 샘플을 돌며 coeffs[9], numSample 누적 ...

    SharedCoeffs[sharedIndex][i] = coeffs[i];  // (3) 자기 슬롯에 부분합 기록
    InterlockedAdd( TotalSample, numSample );  // (4) 샘플 개수는 원자적으로 누적
    GroupMemoryBarrierWithGroupSync();         // (5) 모든 쓰기 완료를 보장

    if ( GI == 0 )                             // (6) 스레드 0이 256개 부분합을 합산
    {
        for ( i ... 256 ) coeffs[j] += SharedCoeffs[i][j];
        float dOmega = 2.f * PI / float( TotalSample );
        Coeffs[i] = coeffs[i] * dOmega;
    }
}
```

## `InterlockedAdd(TotalSample, numSample)` — 경쟁 없는 원자적 누적

- `TotalSample`은 **groupshared** 변수이므로 256개 스레드가 모두 같은 메모리 위치를 본다.
- 각 스레드가 `TotalSample += numSample`처럼 일반 덧셈을 하면 *읽기 → 더하기 → 쓰기* 세 단계 사이에 다른 스레드가 끼어들어 값이 덮어써지는 **race condition(경쟁 조건)** 이 생긴다. 예: 두 스레드가 동시에 0을 읽고 각자 +100을 써 버리면 결과는 200이 아니라 100이 된다.
- `InterlockedAdd`는 하드웨어 원자 연산(atomic add)으로, 읽기-수정-쓰기를 **분할 불가능한 한 번의 연산**으로 수행한다. 따라서 256개 스레드가 동시에 호출해도 최종 `TotalSample`은 정확히 모든 `numSample`의 합이 된다.
- 왜 개수를 따로 세는가: 시작 위치가 스레드마다 다르고 루프 조건이 `< 2π`, `< π` 이므로 스레드별 샘플 수가 미세하게 다를 수 있다. 정확한 총 샘플 수 $n_1 n_2$가 있어야 미소 입체각 `dOmega = 2π / TotalSample`를 맞게 계산할 수 있다(위 수식의 $\frac{2\pi}{n_1 n_2}$ 항).
- 반면 `SharedCoeffs[sharedIndex][i]`는 **스레드마다 고유한 슬롯**(`sharedIndex = GTid.y*16 + GTid.x`)에 쓰므로 충돌이 없고 원자 연산이 필요 없다. 부동소수 `float3`는 HLSL의 `InterlockedAdd`가 지원하지 않는 타입이라는 점도 이런 "슬롯 분리 후 한 스레드가 합산" 설계를 택한 이유다(Interlocked 계열은 int/uint만 지원).

## `GroupMemoryBarrierWithGroupSync` — 쓰기 완료 후 읽기 보장

이 함수는 두 가지를 한 번에 한다.

1. **GroupMemoryBarrier**: 이 지점 이전에 스레드가 수행한 groupshared 메모리 쓰기가 그룹 내 다른 모든 스레드에게 **보이도록(visible)** 만든다.
2. **WithGroupSync**: 그룹의 **모든 스레드가 이 지점에 도달할 때까지** 각 스레드를 대기시킨다(실행 장벽, barrier).

코드에서 두 번 호출되며 각각 다른 위험을 막는다.

| 위치 | 막는 문제 |
|---|---|
| (2) `TotalSample = 0` 직후 | 스레드 0이 아직 0으로 초기화하기 전에 다른 스레드가 `InterlockedAdd`를 해서 초기화가 그 값을 지워 버리는 것 |
| (5) 부분합 기록 + `InterlockedAdd` 직후 | 스레드 0이 아직 다 안 쓰인 `SharedCoeffs`를 읽거나, 일부 스레드의 `numSample`이 아직 더해지지 않은 `TotalSample`로 `dOmega`를 계산하는 것 |

(5)가 없으면 스레드 0은 다른 스레드들이 루프를 마쳤는지 알 길이 없다. 컴퓨트 셰이더에서 한 스레드 그룹의 스레드들은 여러 wave/warp에 걸쳐 있어 실행 순서가 보장되지 않기 때문에, "모두 썼다"는 사실은 명시적 barrier로만 확립된다.

## 두 도구의 역할 분담 요약

| | `InterlockedAdd` | `GroupMemoryBarrierWithGroupSync` |
|---|---|---|
| 대상 | 하나의 groupshared int 변수 | 그룹 내 모든 스레드의 실행·메모리 |
| 해결하는 문제 | **같은 주소**에 대한 동시 쓰기 충돌(값 손실) | **순서** 문제: 쓰기가 끝나기 전에 읽는 것 |
| 비유 | 카운터에 줄 서서 한 명씩 정확히 더함 | 모두 손을 들 때까지 기다린 뒤 다음 단계로 |
| 이 코드에서 | 총 샘플 개수 `TotalSample` 집계 | 초기화 완료 보장 + 256개 부분합·카운트 완료 보장 |

핵심 한 줄: **`InterlockedAdd`는 "정확한 합"을, `GroupMemoryBarrierWithGroupSync`는 "합산 시점"을 보장한다.** 둘이 함께 있어야 스레드 0이 올바른 `SharedCoeffs`와 `TotalSample`로 최종 SH 계수 `Coeffs[9]`를 쓸 수 있다.

## 참고: 자주 하는 실수

- `GroupMemoryBarrier()`(Sync 없는 버전)만 쓰면 메모리 가시성만 보장하고 스레드가 해당 지점에 도착했는지는 보장하지 않으므로 이 코드에서는 부족하다.
- barrier는 **분기 안에서 일부 스레드만** 호출하면 안 된다(전 스레드가 같은 barrier에 도달해야 함). 원문 코드도 barrier를 `if (GI == 0)` 블록 바깥에 두었다.
- `InterlockedAdd`는 스레드 그룹 간(다른 group 사이)에는 groupshared 변수로 동작하지 않는다. 그룹 간 합산이 필요하면 UAV 버퍼에 대한 Interlocked 연산을 써야 한다. 이 셰이더는 그룹 하나(256 스레드)로 큐브맵 전체를 적분하므로 groupshared로 충분하다.
