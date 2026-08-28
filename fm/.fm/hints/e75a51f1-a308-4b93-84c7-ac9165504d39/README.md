# $L_{lm}$ 컴퓨트 셰이더의 스레드 그룹 구성과 작업 분배

## 질문
$L_{lm}$을 구하는 컴퓨트 셰이더의 스레드 그룹 구성과 작업 분배 방식은?

## 답
`[numthreads(16, 16, 1)]`로 **16×16 = 256 스레드**짜리 하나의 스레드 그룹을 사용한다. 각 스레드는
- $\phi$: `GTid.x * SampleDelta`에서 시작해 `DeltaPhi = SampleDelta * 16` 간격으로,
- $\theta$: `GTid.y * SampleDelta`에서 시작해 `DeltaTheta = SampleDelta * 16` 간격으로

구면을 훑으며 샘플을 나눠 맡는다(스트라이드 분할). 마지막에 그룹 공유 메모리로 256개의 부분합을 모아 최종 계수 9개를 쓴다.

---

## 1. 계산하려는 식

원문은 2차(L2) 구면 조화 계수 9개를 큐브맵에서 투영해 구한다.

$$ L_{lm}=\frac{1}{\pi}\int_{0}^{2\pi}\!\!\int_{0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi
\;\approx\; \frac{2\pi}{n_1 n_2}\sum_{\phi}\sum_{\theta} L\,y_l^m\,\sin\theta $$

즉 $(\phi,\theta)$ 격자 위의 **모든 샘플에 대한 합**이므로, 샘플들을 여러 스레드에 나눠 계산하고 합치기만 하면 되는 전형적인 병렬 리덕션 문제다.

## 2. 스레드 그룹 구성

```hlsl
static const int ThreadGroupX = 16;
static const int ThreadGroupY = 16;

static const float SampleDelta = 0.025f;
static const float DeltaPhi   = SampleDelta * ThreadGroupX;   // 0.4
static const float DeltaTheta = SampleDelta * ThreadGroupY;   // 0.4

groupshared float3 SharedCoeffs[ThreadGroupX * ThreadGroupY][9];
groupshared int    TotalSample;

[numthreads(ThreadGroupX, ThreadGroupY, 1)]
void main( uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
```

- `numthreads(16,16,1)`: 그룹 하나에 X 16 × Y 16 = 256 스레드. 디스패치는 그룹 1개만 하면 된다(출력이 계수 9개뿐이므로).
- `SV_GroupThreadID`(`GTid`): 그룹 안에서의 2D 좌표 (0..15, 0..15). 이 값이 각 스레드의 **시작 오프셋**이 된다.
- `SV_GroupIndex`(`GI`): 0..255 선형 인덱스. `GI == 0`인 스레드가 초기화와 최종 합산을 담당한다.
- `groupshared`: 그룹 공유 메모리. 스레드별 부분합 256×9개와 총 샘플 수를 담는다.

## 3. 작업 분배 — 인터리브(스트라이드) 분할

```hlsl
for ( float phi = GTid.x * SampleDelta; phi < 2.f * PI; phi += DeltaPhi )
{
    for ( float theta = GTid.y * SampleDelta; theta < PI; theta += DeltaTheta )
    {
        float3 sampleDir = normalize( float3( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) ) );
        float3 radiance  = CubeMap.SampleLevel( LinearSampler, sampleDir, 0 ).rgb;
        float y[9];  ShFunctionL2( sampleDir, y );
        for ( int i = 0; i < 9; ++i ) coeffs[i] += radiance * y[i] * sin( theta );
        ++numSample;
    }
}
```

전체 격자를 `SampleDelta = 0.025` 간격으로 잡으면 $\phi$ 방향 약 252칸, $\theta$ 방향 약 126칸이다. 이걸 256 스레드가 나누는 방식은 다음과 같다.

| | 시작점 | 간격 | 의미 |
|---|---|---|---|
| $\phi$ | `GTid.x * 0.025` | `DeltaPhi = 0.4` (16칸) | 스레드 x가 격자 열 `x, x+16, x+32, …`를 맡음 |
| $\theta$ | `GTid.y * 0.025` | `DeltaTheta = 0.4` (16칸) | 스레드 y가 격자 행 `y, y+16, y+32, …`를 맡음 |

즉 격자 인덱스 $(i, j)$의 샘플은 스레드 $(i \bmod 16,\; j \bmod 16)$이 처리한다. 격자를 16×16 타일로 덮어 놓고, 각 타일의 같은 위치를 같은 스레드가 담당하는 그림이다.

- 연속 블록으로 쪼개는 대신 **stride 16으로 건너뛰며** 훑기 때문에 모든 스레드가 거의 같은 개수(~126개)의 샘플을 맡아 부하가 균등하다.
- 시작점을 `GTid * SampleDelta`로, 간격을 `SampleDelta * 그룹크기`로 잡으면 스레드끼리 **중복도 누락도 없이** 격자를 정확히 분할하게 된다.
- 픽셀 셰이더 버전의 Irradiance Map(앞부분 코드)이 픽셀당 전체 격자를 직렬로 돌던 것과 대비된다 — 여기서는 출력이 계수 9개뿐이라 픽셀 병렬성이 없으므로, 대신 **샘플을 스레드에 분산**한다.

## 4. 합산(리덕션) 단계

```hlsl
int sharedIndex = GTid.y * ThreadGroupX + GTid.x;
for i: SharedCoeffs[sharedIndex][i] = coeffs[i];   // 부분합 저장
InterlockedAdd( TotalSample, numSample );           // 총 샘플 수 원자적 누적
GroupMemoryBarrierWithGroupSync();

if ( GI == 0 )
{
    for ( int i = 0; i < 256; ++i ) for j: coeffs[j] += SharedCoeffs[i][j];
    float dOmega = 2.f * PI / float( TotalSample );  // 2π/(n1·n2)
    for i: Coeffs[i] = coeffs[i] * dOmega;
}
```

1. 각 스레드가 자기 부분합 9개를 `SharedCoeffs[GTid.y*16 + GTid.x]`에 쓰고, 처리한 샘플 수를 `TotalSample`에 `InterlockedAdd`한다.
2. `GroupMemoryBarrierWithGroupSync()`로 256 스레드가 모두 쓰기를 끝냈음을 보장한다(시작 부분에서 `TotalSample = 0` 초기화 뒤에도 같은 배리어가 있다).
3. `GI == 0` 스레드 하나가 256개 부분합을 직렬로 더하고, $d\Omega = 2\pi / \text{TotalSample}$를 곱해 위 식의 $\frac{2\pi}{n_1 n_2}$ 정규화를 적용한 뒤 `RWStructuredBuffer Coeffs`(u0)에 9개 계수를 쓴다.

## 5. 정리

- **구성**: `numthreads(16,16,1)` → 그룹 1개, 256 스레드, `groupshared`로 리덕션.
- **분배**: `GTid.x`/`GTid.y`를 시작 오프셋, `SampleDelta*16`을 stride로 하여 $(\phi,\theta)$ 격자를 인터리브 분할.
- **결과**: 스레드별 부분합 → 공유 메모리 → 스레드 0이 합산·정규화 → $L_{lm}$ 9개.
