# 반구 적분을 구 전체 적분으로 바꿔도 결과가 같은 이유

## 한 줄 답
피적분 함수에 **$\max(\cos\theta, 0)$** (clamped cosine)을 곱해 두면, 법선 반대편 반구($\theta > \pi/2$)에서는 $\cos\theta < 0$이라 이 항이 0이 되어 기여가 사라진다. 따라서 적분 범위를 $\theta \in [0, \pi]$(구 전체)로 넓혀도 실제로 남는 것은 $\theta \in [0, \pi/2]$(반구)의 Radiance만이다.

## 원래 식: 반구 위의 Irradiance
Lambertian(Diffuse) 표면의 Irradiance는 표면 법선 $\mathbf n$을 중심으로 한 **상반구** $\Omega$에서 들어오는 Radiance를 코사인 가중으로 적분한 것이다.

$$
E = \frac{1}{\pi}\int_{\phi=0}^{2\pi}\int_{\theta=0}^{\pi/2} L_i(\theta,\phi)\,\cos\theta\,\sin\theta\,d\theta\,d\phi
$$

- $\cos\theta = \omega_i \cdot \mathbf n$: 빛이 비스듬히 들어올수록 단위 면적당 받는 에너지가 줄어드는 항
- $\sin\theta\,d\theta\,d\phi$: 구면 좌표에서의 입체각 미소 요소 $d\omega$
- 적분 상한이 $\pi/2$인 이유: 표면 뒤쪽($\theta > \pi/2$)에서 오는 빛은 표면에 도달하지 못하므로 애초에 세지 않는다.

## 왜 굳이 구 전체로 바꾸는가?
구면 조화 함수(SH)는 **구 전체 $S^2$ 위에서 정의된 직교 기저**다. SH 계수를 구하는 투영 $L_{lm} = \int_{S^2} L\,y_l^m\,d\omega$ 역시 구 전체 적분이다. 그런데 원래 Irradiance 식은 반구 적분이라 그대로는 SH 투영 형태에 맞지 않는다. 또 반구의 방향은 법선 $\mathbf n$마다 달라지므로, 법선별로 적분 범위를 바꿔 계산하는 것은 비효율적이다.

그래서 원문은 식을 두 부분으로 분리한다.

1. **Radiance 부분** — 적분 범위를 구 전체로 확장
   $$\int_{0}^{2\pi}\int_{0}^{\pi} L_i(\theta,\phi)\,\sin\theta\,d\theta\,d\phi$$
2. **코사인 부분** — 양수 범위로 클램프
   $$\max(\cos\theta, 0)$$

두 부분을 곱한 뒤 구 전체에서 적분하면 원래 반구 적분과 동일하다.

## 왜 결과가 같은지 — 식으로 확인
구 전체 적분을 두 구간으로 쪼개면

$$
\int_{0}^{\pi} L\,\max(\cos\theta,0)\,\sin\theta\,d\theta
= \underbrace{\int_{0}^{\pi/2} L\,\cos\theta\,\sin\theta\,d\theta}_{\theta \le \pi/2:\ \max(\cos\theta,0)=\cos\theta}
+ \underbrace{\int_{\pi/2}^{\pi} L\cdot 0\cdot\sin\theta\,d\theta}_{\theta > \pi/2:\ \cos\theta<0 \Rightarrow \max(\cdot,0)=0}
$$

두 번째 항이 정확히 0이므로 첫 번째 항, 즉 원래의 반구 적분만 남는다. 적분 범위를 늘린 것이 아니라 **적분 범위는 늘리되 가중 함수가 그 늘어난 부분을 0으로 죽이는** 구조다.

직관적으로 말하면 $\max(\cos\theta,0)$은 "이 방향에서 오는 빛이 표면 앞쪽인가?"를 판정하는 **마스크**이면서 동시에 코사인 가중치 역할까지 하는 함수다. 반구 제한을 적분 *범위*가 아닌 *피적분 함수*에 녹여 넣은 것이다.

## 이렇게 바꾸면 얻는 것
이제 Radiance와 clamped cosine을 각각 독립적으로 SH에 투영할 수 있다.

$$
L_{lm} = \int_{0}^{2\pi}\int_{0}^{\pi} L(\theta,\phi)\,y_l^m(\theta,\phi)\,\sin\theta\,d\theta\,d\phi,
\qquad
A_l = \int_{0}^{\pi} \max(\cos\theta,0)\,y_l^0(\theta,0)\,d\theta
$$

- $L_{lm}$: 환경맵(조명)만의 SH 계수 — 법선과 무관하게 한 번만 계산
- $A_l$: clamped cosine의 SH 계수 — $\theta$(천정각)에만 의존하는 회전 대칭 함수이므로 $m=0$ 성분만 존재하며, 조명과 무관한 **상수**
- 두 함수의 구면 위 컨볼루션은 SH 공간에서 계수의 곱이 되므로 Irradiance 계수는 $E_{lm} = \sqrt{\tfrac{4\pi}{2l+1}}\,A_l\,L_{lm}$ 로 즉시 얻어진다. 여기서 $\sqrt{4\pi/(2l+1)}$은 로컬 좌표($z$축 = 법선)에서 계산한 $A_l$을 임의 법선 방향으로 회전시키기 위한 가중치다.

![clamped cosine의 SH 계수 A_l (Ramamoorthi & Hanrahan)](fig-1.png)

위 그림은 $\max(\cos\theta,0)$을 SH에 투영해 얻은 계수 $A_l$을 차수 $l$에 대해 그린 것이다. $l=0,1$에서 가장 크고 $l=2$에서 절반 정도로 떨어진 뒤, $l=3$은 정확히 0, 이후 홀수 차수는 모두 0이고 짝수 차수만 부호를 바꾸며 빠르게 감소한다($\hat A_4 \approx -0.13$, $\hat A_6 \approx 0.05$). $\theta = \pi/2$에서 꺾이는 clamped cosine은 매우 부드러운(저주파) 함수이므로 $l \le 2$, 즉 9개 계수(RGB면 27개)만으로 Irradiance Map을 충분히 근사할 수 있다는 것이 이 그래프의 결론이다. 이 저주파 특성 덕분에 반구 마스킹을 $\max(\cos\theta,0)$ 형태로 피적분 함수에 넣어 구 전체에서 SH 투영하는 전략이 실용적으로 성립한다.

## 정리
| | 원래 식 | 변형 식 |
|---|---|---|
| 적분 범위 | 반구 $\theta\in[0,\pi/2]$ | 구 전체 $\theta\in[0,\pi]$ |
| 코사인 항 | $\cos\theta$ | $\max(\cos\theta,0)$ |
| 뒷반구 기여 | 범위에서 제외 | 가중치 0으로 소거 |
| 결과 | 동일 | 동일 |
| 장점 | — | SH 투영 형태에 맞음, $L_{lm}$과 $A_l$을 분리해 사전 계산 가능 |

## 참고
- 원문: `spherical-harmonics.md` — "구면 조화 함수를 이용한 Irradiance Map" 절
- Ramamoorthi & Hanrahan, *On the relationship between radiance and irradiance* (JOSA 2001), 4.A
- Green, *Spherical Harmonic Lighting: The Gritty Details* (2003)
