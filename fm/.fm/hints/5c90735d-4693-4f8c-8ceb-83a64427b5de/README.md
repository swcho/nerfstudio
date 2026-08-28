# 큐브맵 6개 면의 Irradiance Map을 한 번의 드로우 콜로 생성하기

## Q
큐브맵 6개 면에 대한 Irradiance Map을 한 번의 드로우 콜로 생성하는 방법은?

## A (요약)
1. Irradiance Map용 큐브맵 텍스처(32x32, 6면)를 **렌더 타겟 바인딩 가능**하게 만들어 렌더 타겟으로 바인딩한다.
2. 버텍스 버퍼/인덱스 버퍼 없이 **버텍스 수 6**으로 드로우 콜을 호출한다. 버텍스 셰이더는 `SV_VertexID`(0~5 = 면 인덱스)를 그대로 넘긴다.
3. **지오메트리 셰이더**가 각 point 입력마다 정육면체의 한 면(사각형 = 4 버텍스 triangle strip)을 생성하고, 출력 구조체의 `SV_RenderTargetArrayIndex`에 면 인덱스를 써서 **어느 큐브맵 면(텍스처 배열 슬라이스)에 그릴지 지정**한다.
4. 픽셀 셰이더는 면 위의 `localPosition`을 노멀 방향으로 삼아 반구 적분(리만합)으로 irradiance를 계산한다.

---

## 1. 배경: Irradiance Map이 왜 큐브맵인가
Irradiance Map은 이미지를 광원으로 쓰는 IBL 중 Diffuse 반사를 위해 사전 필터링한 환경맵이다. 전 방향(360도)의 빛을 담아야 하므로 큐브맵을 쓴다.

![원본 환경 큐브맵(왼쪽)과 그것을 컨볼브한 Irradiance Map(오른쪽)](fig-1.png)

그림(learnopengl 출처): 왼쪽은 원본 환경 큐브맵을 십자 형태로 펼친 것이고, 오른쪽은 같은 큐브맵을 반구 적분해 얻은 Irradiance Map이다. 오른쪽은 디테일이 전부 사라지고 대략적 색/밝기 분포만 남아 있는데, 이것이 "저주파이므로 32x32처럼 작은 해상도로 충분하다"는 근거다.

## 2. 문제: 큐브맵의 6면을 어떻게 한 번에 렌더링하나
큐브맵 한 면마다 렌더 타겟을 바꿔 6번 드로우하는 단순한 방법 대신, 원문 구현(Direct3D 11/12)은 다음 조합으로 **드로우 콜 1회**로 끝낸다.

- 큐브맵은 D3D에서 **6 슬라이스 텍스처 배열**이다. RTV를 배열 전체(ArraySize=6)로 만들면 한 번에 바인딩할 수 있다.
- 어느 슬라이스에 픽셀을 쓸지는 **지오메트리 셰이더**가 출력하는 시스템 값 시맨틱 `SV_RenderTargetArrayIndex`가 결정한다. (D3D11에서 이 시맨틱은 GS 출력에서만 쓸 수 있고, PS에는 입력으로 전달된다. D3D12/최신 하드웨어에서는 `VPAndRTArrayIndexFromAnyShaderFeedingRasterizer` 옵션이 있으면 VS에서도 쓸 수 있지만, 원문은 GS 방식이다.)

### 2-1. 렌더 타겟 준비
```cpp
agl::TextureTrait trait = cubeMap->GetTrait();
trait.m_width = trait.m_height = 32;                      // 원본 2048 -> 32
trait.m_format = agl::ResourceFormat::R8G8B8A8_UNORM_SRGB;
trait.m_bindType |= agl::ResourceBindType::RenderTarget;  // 렌더 타겟으로 사용
auto irradianceMap = agl::Texture::Create( trait );
```
원본 큐브맵의 trait을 복사하므로 "큐브맵(6면 배열)" 속성은 유지되고, 크기와 바인딩 플래그만 바뀐다.

### 2-2. 버텍스 셰이더: 면 인덱스만 넘긴다
```hlsl
struct VS_OUTPUT { uint vertexId : VERTEXID; };
VS_OUTPUT main( uint vertexId : SV_VertexID )
{
    VS_OUTPUT output = (VS_OUTPUT)0;
    output.vertexId = vertexId;   // 0..5 = 큐브맵 면 번호
    return output;
}
```
버텍스/인덱스 버퍼를 바인딩하지 않고 `Draw(6)`처럼 **버텍스 6개**만 요청하면, `SV_VertexID`가 0~5로 들어온다. 이 값이 곧 "몇 번째 면을 그릴 것인가"다. 실제 정육면체 지오메트리는 여기서 만들지 않는다.

### 2-3. 지오메트리 셰이더: 면을 생성하고 슬라이스를 지정
```hlsl
struct GS_OUTPUT
{
    float4 position      : SV_POSITION;
    float3 localPosition : POSITION0;
    uint   rtIndex       : SV_RenderTargetArrayIndex;   // 핵심
};

static const float4 projectedPos[] = { {-1,-1,0,1}, {-1,1,0,1}, {1,-1,0,1}, {1,1,0,1} }; // 화면 전체 사각형
static const float3 vertices[]     = { /* 정육면체 8개 꼭짓점 (-1..1) */ };
static const int4   indices[]      = { {6,7,2,3}, {0,1,4,5}, {5,1,7,3}, {0,4,2,6}, {4,5,6,7}, {2,3,0,1} }; // 면별 꼭짓점 4개

[maxvertexcount(4)]
void main( point GS_INPUT input[1], inout TriangleStream<GS_OUTPUT> triStream )
{
    GS_OUTPUT output = (GS_OUTPUT)0;
    output.rtIndex = input[0].vertexId;             // 이 사각형은 vertexId번 면에 그린다
    for ( int i = 0; i < 4; ++i )
    {
        output.position      = projectedPos[i];     // NDC 전체를 덮는 풀스크린 쿼드
        output.localPosition = vertices[ indices[input[0].vertexId][i] ]; // 그 면의 3D 위치
        triStream.Append( output );
    }
    triStream.RestartStrip();
}
```
포인트 하나(=면 하나)를 입력받아 **4개 버텍스의 triangle strip(사각형)** 을 출력한다. 두 가지 좌표를 동시에 내보내는 점이 핵심이다.

| 출력 | 역할 |
|---|---|
| `position` (`SV_POSITION`) | 항상 NDC의 (-1,-1)~(1,1). 즉 렌더 타겟 슬라이스 전체(32x32)를 덮는다. |
| `localPosition` | 정육면체 해당 면의 실제 꼭짓점 좌표. 래스터라이저가 보간해 주면 픽셀마다 "큐브 표면 위의 방향 벡터"가 된다. |
| `rtIndex` (`SV_RenderTargetArrayIndex`) | 이 프리미티브를 배열의 몇 번째 슬라이스(+X, -X, +Y, -Y, +Z, -Z)에 래스터화할지 지정. |

`indices` 표는 D3D 큐브맵 면 순서(0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z)에 맞춰 각 면의 4개 꼭짓점을 고른 것이다. 예: 면 0 = `{6,7,2,3}` = x가 모두 +1인 꼭짓점 → +X 면.

결과적으로 GS 한 번의 실행이 6번 일어나고(입력 포인트 6개), 각 실행이 서로 다른 슬라이스에 풀스크린 쿼드 하나씩을 그리므로, **드로우 콜 1회로 6면이 모두 채워진다**.

### 2-4. 픽셀 셰이더: 방향 벡터로 반구 적분
```hlsl
float3 normal = normalize( input.localPosition );   // 이 픽셀이 대표하는 방향
// normal 기준 탄젠트 프레임(right, up, normal) 구성 후
for phi in [0, 2pi), for theta in [0, pi/2):
    irradiance += CubeMap.Sample( LinearSampler, worldSample ).rgb * cos(theta) * sin(theta);
irradiance = PI * irradiance / numSample;
```
GS가 넘긴 `localPosition`을 정규화하면 큐브맵 해당 텍셀의 방향이 된다. 그 방향을 노멀로 하는 반구에서 원본 큐브맵을 샘플링해 리만합
$\frac{\pi}{n_1 n_2}\sum\sum L_i\cos\theta\sin\theta$ 를 계산한다(Lambert BRDF의 $1/\pi$가 포함된 형태, 알베도는 렌더링 시 곱함).

## 3. 결과

| 원본 스카이박스 | Irradiance Map |
|---|---|
| ![원본 큐브맵](fig-2.png) | ![Irradiance Map](fig-3.png) |

왼쪽 원본 큐브맵에는 구름, 수평선, 바다의 질감이 뚜렷하다. 오른쪽 결과는 같은 6면 십자 배치이지만 32x32로 작고 완전히 흐려져, 위쪽 면은 어두운 남색(하늘 천정), 가운데 띠는 밝은 청회색(수평선 근처), 아래 면은 짙은 파랑(바다)만 남았다. 6면 모두 한 번의 드로우 콜로 이렇게 채워진다. 이 텍스처는 `IrradianceMap.Sample(LinearSampler, normal)`로 조명 계산에 사용된다.

## 4. 기억할 포인트
- **렌더 타겟 = 큐브맵 자체**(6 슬라이스 배열 RTV) + `RenderTarget` 바인드 플래그.
- **드로우 콜은 `Draw(6)`**, 버퍼 없음, `SV_VertexID`가 면 번호.
- **GS가 면 사각형 생성**, `SV_RenderTargetArrayIndex = vertexId`로 슬라이스 선택.
- 이 방식은 셰도우 큐브맵(포인트 라이트 옴니 셰도우), 큐브맵 리플렉션 캡처 등 "큐브맵 6면을 한 패스에 그리기"의 일반적인 D3D 패턴이다.
- 메모리: 32*32*6*4 B ≈ 24KB. 이후 원문은 이를 SH 9계수(108 B)로 대체하는 최적화를 다룬다.

## 참고
- 원문: 구면 조화 함수(Spherical Harmonics) 글, "Irradiance Map" 절
- <https://learnopengl.com/PBR/IBL/Diffuse-irradiance>
- Microsoft Docs, System-Value Semantics: `SV_RenderTargetArrayIndex`
- 코드: <https://github.com/xtozero/ssr/tree/irradiance_map>
