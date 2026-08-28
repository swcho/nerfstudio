# Irradiance Map이란 무엇인가?

**한 줄 답**: 이미지를 광원으로 쓰는 **IBL(Image Based Lighting)** 의 한 종류로, **Diffuse(난반사) 조명을 위해 환경 이미지를 미리(사전에) 필터링·적분해 둔 이미지**다. 360도 전 방향을 담아야 하므로 대체로 **큐브맵(cubemap)** 형태로 저장한다.

## 1. IBL과 Irradiance Map의 관계

- **IBL**: 점광원·방향광 같은 해석적 광원 대신, 주변 환경을 찍은 이미지(스카이박스, 환경맵)의 픽셀 하나하나를 광원으로 취급하는 조명 기법.
- IBL은 보통 두 갈래로 나뉜다.
  - **Diffuse IBL** → **Irradiance Map** (이 카드의 주제)
  - **Specular IBL** → Pre-filtered environment map + BRDF LUT (거칠기별 반사)
- 즉 Irradiance Map은 "IBL 중 Diffuse 항을 담당하는, 미리 계산해 둔 결과물"이다.

![원본 환경 큐브맵(좌)과 그것을 Diffuse용으로 사전 필터링한 Irradiance Map(우)](fig-1.png)

위 그림(learnopengl.com 출처)에서 왼쪽은 원본 환경 큐브맵을 십자 형태로 펼친 것이고, 오른쪽은 같은 큐브맵으로부터 만든 Irradiance Map이다. 나무·길 같은 세부 디테일이 모두 사라지고 **방향에 따라 부드럽게 변하는 색 덩어리**만 남는 것이 관찰된다. 하늘 쪽(위)은 밝고 푸르스름하며, 땅 쪽(아래)은 어둡고 갈색빛이다. 즉 Irradiance Map의 각 픽셀은 "그 방향을 법선으로 갖는 표면이 반구 전체에서 받는 빛의 총량"을 뜻한다.

## 2. 왜 "사전에 필터링"하는가 — 렌더링 방정식

Diffuse 표면이 받는 빛은 렌더링 방정식에서 시작한다.

$$L_r(\mathbf{x},\omega_o) = L_e + \int_\Omega f_r(\mathbf{x},\omega_i,\omega_o)\,L_i(\mathbf{x},\omega_i)\,(\omega_i\cdot\mathbf{n})\,d\omega_i$$

- Diffuse에서는 BRDF가 **Lambert BRDF** $\sigma/\pi$ 로 상수이므로 적분 밖으로 빼낼 수 있고,
- 표면 자체 발광 $L_e = 0$ 이라고 두면

$$L_r = \frac{\sigma}{\pi}\underbrace{\int_\Omega L_i(\omega_i)(\omega_i\cdot\mathbf{n})\,d\omega_i}_{E(\mathbf{n})\ =\ \text{Irradiance}}$$

- 남은 적분 $E(\mathbf{n})$이 **Irradiance**이며, **법선 $\mathbf{n}$ 하나에만 의존**한다(뷰 방향 $\omega_o$와 무관).
- 따라서 "법선 방향 → Irradiance" 룩업 테이블을 만들 수 있고, 이것을 큐브맵에 저장한 것이 Irradiance Map이다. 매 픽셀마다 반구 적분을 실시간으로 하는 대신, 오프라인/초기화 시점에 한 번만 계산해 둔다("사전 필터링").

## 3. 왜 큐브맵인가

- 법선 벡터 $\mathbf{n}$은 단위 구면 위의 임의 방향이므로, **전 방향(360도)** 을 빠짐없이 커버하는 텍스처가 필요하다.
- 큐브맵은 3D 방향 벡터로 직접 샘플링(`TextureCube.Sample(sampler, normal)`)할 수 있어 GPU에서 가장 자연스럽다.
- Diffuse 응답은 **저주파**이므로 해상도가 매우 작아도 된다. 원문에서는 2048x2048 원본 큐브맵에서 **32x32 큐브맵**(약 24KB = 32·32·6·4B)으로 Irradiance Map을 만들고, 텍스처 선형 보간으로 충분한 품질을 얻는다.

## 4. 원문의 실제 생성 결과

| 원본 스카이박스 | Irradiance Map |
| --- | --- |
| ![원본 스카이박스 큐브맵](fig-2.png) | ![생성된 32x32 Irradiance Map](fig-3.png) |

- 왼쪽(fig-2): 구름 낀 하늘과 바다 수평선이 선명한 원본 큐브맵.
- 오른쪽(fig-3): 같은 장면을 반구 적분해 얻은 Irradiance Map. 구름·수평선 디테일이 모두 사라지고, 위쪽(하늘을 향하는 법선)은 어두운 남색, 옆면은 수평선의 밝은 빛이 섞여 상대적으로 밝은 청회색, 아래쪽(바다를 향하는 법선)은 짙은 청색으로 부드럽게 그라데이션된다. 32x32 해상도라 약간의 블록 느낌이 보이지만 Diffuse 조명에는 충분하다.

## 5. 생성과 사용 방식(원문 요약)

- **생성**: 32x32 큐브맵을 렌더 타겟으로 잡고, 지오메트리 셰이더의 `SV_RenderTargetArrayIndex`로 6면을 한 번의 드로우 콜에 그린다. 픽셀 셰이더는 각 출력 방향을 법선으로 두고 반구를 (theta, phi)로 균일 샘플링하여 `CubeMap.Sample(...) * cos(theta) * sin(theta)`를 누적한 뒤 `PI * sum / numSample`로 정규화한다.
- **사용**: 셰이딩 시 표면 법선으로 한 번만 샘플링한다.
  ```hlsl
  float3 ImageBasedLight(float3 normal) { return IrradianceMap.Sample(LinearSampler, normal).rgb; }
  lightColor = ImageBasedLight(normal) * Diffuse;   // Lambert의 sigma/pi는 알베도 쪽에 포함
  ```

## 6. 다음 단계와의 연결 (구면 조화 함수)

원문의 핵심 동기: Irradiance는 저주파이므로 큐브맵(24KB) 대신 **구면 조화 함수(SH) 계수 27개(RGB 3 × 9계수, 108Byte)** 만으로 거의 동일하게 근사할 수 있다(Ramamoorthi & Hanrahan, 2001). 즉 Irradiance Map은 SH 기반 조명 표현이 대체하려는 "기준 표현"이다.

## 암기 포인트

1. IBL의 일종 — 이미지가 광원.
2. **Diffuse** 전용 — Lambert BRDF가 상수라 적분 밖으로 나가고, 남은 적분이 Irradiance $E(\mathbf{n})$.
3. **사전 필터링** — 법선 방향별 반구 적분을 미리 계산해 텍스처로 저장.
4. **큐브맵** — 전 방향 커버 + 법선 벡터로 직접 샘플링, 저주파라 32x32로도 충분.
