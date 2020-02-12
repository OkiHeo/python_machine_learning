지도 학습의 문제는 '회귀'와 '분류'의 문제로 나눌 수 있다.
지도학습 : 훈련 데이터로부터 하나의 함수를 유추해내기 위한 머신러닝의 한 방법.

* 회귀 : 입력에 대해 연속적인 값을 대응시키는 문제
* 분류 : 입력에 대해 순서가 없는 클래스(라벨)를 대응시키는 문제

<br>

## 5.1 1차원 입력 직선 모델

나이 __x__ 와 키 __t__ 가 세트로 된 데이터가 있다.
16인분의 데이터가 있다고 생각하면 16*1벡터로 나타낼 수 있다.

데이터베이스에 없는 사람의 나이에 대해 그 사람의 키를 예측해보자.
먼저, 나이와 몸무게의 인공 데이터를 만든다.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 데이터 생성
np.random.seed(seed=1)    # 난수를 고정하기 위해
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
# 16개의 랜덤 나이 생성
X = 5 + 25*np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0]-Prm_c[1]*np.exp(-Prm_c[2]*X)+4*np.random.randn(X_n)
# 생성한 데이터를 파일로 저장한다.
np.savez('ch5_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T)

print(np.round(X,2))
print(np.round(T,2))
====
[15.43 23.01  5.   12.56  8.67  7.31  9.66 13.64 14.92 18.47 15.48 22.13
 10.11 26.95  5.68 21.76]
[170.91 160.68 129.   159.7  155.46 140.56 153.65 159.43 164.7  169.65
  160.71 173.29 159.31 171.52 138.96 165.87]
```

```python
# 데이터 그래프
plt.figure(figsize=(4, 4))
plt.plot(X, T, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
====

```

<br>

### 5.1.1 직선 모델

데이터가 고르지 않아서, 새로운 나이 데이터에 키를 '정확히' 맞추는 것은 불가능하다.
**어느 정도 오차를 허용** 한다면, 주어진 데이터에서 직선을 긋는 것으로 그럴듯하게 예측 가능하다.

y(x) = w0*x + w1
기울기를 나타내는 w0와 y절편을 나타내는 w1에 적당한 값을 넣으면 데이터에 부합한 직선을 만들 수 있다.

<br>

### 5.1.2 제곱 오차 함수

'데이터에 부합하도록' 오차 J를 정의한다.

J는 평균 제곱 오차(mean square erre, MSE)로 직선과 데이터 점의 차의 제곱의 평균이다.
(오차의 크기가 N에 의존하지 않는다.)

어떤 w0과 w1를 선택하더라도, 위의 그래프에서 볼 수 있듯이 데이터가 직선 상에 나란히 있지 않기 때문에 J가 0이 되지는 않는다.

w와 J의 관계를 그래프로 나타낸다면, 아래와 같다.

w공간에서의 평균 제곱 오차는 계곡과 같은 모양을 하고 있다.
w0의 방향의 변화에 J가 크게 변화한다. 기울기가 조금이라도 변하면 직선이 데이터 점에서 크게 어긋나기 때문.
계곡의 바닥도 절편 w1방향으로 높이가 변화한다.

### 5.1.3 매개 변수 구하기(경사 하강법)
경사 하강법(steepest descent method) : 1차 근삿값 발견용 최적화 알고리즘.
기본 아이디어는 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지 반복시키는 것.

w0와 w1는 J위의 한 지점에 대응한다.
J를 w0와 w1로 편미분한 벡터를 J의 기울기라고 하며,
∇wJ로 나타낸다. __J를 최소화하려면 J의 기울기의 반대 방향(-∇wJ)으로 진행__ 하면 좋다.

∴ w의 갱신 방법(학습 법칙)을 행렬 표기로 나타내면
w(t+1) = w(t) -α∇wJ|\_{w(t)}
α : 학습률. w 갱신의 폭을 조절한다.

기울기를 계산하는 함수 __`dmse_line(x, t, w)`__ 만든다.
인수 데이터 x, t, w를 전달하면 w의 기울기 d_w0, d_w1를 리턴한다.

```python
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y - t) * x)
    d_w1 = 2 * np.mean(y - t)
    return d_w0, d_w1
```

```python
d_w = dmse_line(X, T, [10, 165])
print(np.round(d_w, 1))
====
[5046.3  301.8]
```

이는 차례로 w0과 w1 방향의 기울기를 나타낸다.

<br>

`dmse_line`를 사용한 경사 하강법 __`fit_line_num(x, t)`__ 를 구현한다.
데이터 x, t를 인수로 하여 mse_line을 최소화하는 w를 리턴해준다.
갱신 단계의 폭이 되는 학습률 α는 0.001로 한다.

```python
# 경사 하강법 ------------------------------------
def fit_line_num(x, t):
    w_init = [10.0, 165.0] # 초기 매개 변수
    alpha = 0.001 # 학습률
    i_max = 100000 # 반복의 최대 수
    eps = 0.1 # 반복을 종료 기울기의 절대 값의 한계
    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init
    for i in range(1, i_max):
        dmse = dmse_line(x, t, w_i[i - 1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        if max(np.absolute(dmse)) < eps: # 종료판정, np.absolute는 절대치
            break
    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    w_i = w_i[:i, :]
    return w0, w1, dmse, w_i


# 메인 ------------------------------------
plt.figure(figsize=(4, 4)) # MSE의 등고선 표시
xn = 100 # 등고선 해상도
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))
for i0 in range(xn):
    for i1 in range(xn):
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))
cont = plt.contour(xx0, xx1, J, 30, colors='black',
                   levels=(100, 1000, 10000, 100000))
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)

# 경사 하강법 호출
W0, W1, dMSE, W_history = fit_line_num(X, T)

# 결과보기
print('반복 횟수 {0}'.format(W_history.shape[0]))
print('W=[{0:.6f}, {1:.6f}]'.format(W0, W1))
print('dMSE=[{0:.6f}, {1:.6f}]'.format(dMSE[0], dMSE[1]))
print('MSE={0:.6f}'.format(mse_line(X, T, [W0, W1])))
plt.plot(W_history[:, 0], W_history[:, 1], '.-',
         color='gray', markersize=10, markeredgecolor='cornflowerblue')
plt.show()
```

기울기가 적당히 평평해지면 for문에 빠져나오도록 했다.

계곡의 중앙 부근에서 기울기가 거의 없어지는 지점에 도달하게 된다.(푸른색이 w의 갱신된 모습)

<br>

갱신된 w0와 w1를 직선모델에 대입시켜 데이터 위에 겹쳐보면, 적당한 위치에 위치하는 것을 볼 수 있다.

```python
# 선 표시 ----------------------------------
def show_line(w):
    xb = np.linspace(X_min, X_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)


# 메인 ------------------------------------
plt.figure(figsize=(4, 4))
W=np.array([W0, W1])
mse = mse_line(X, T, W)
print("w0={0:.3f}, w1={1:.3f}".format(W0, W1))
# mse = mse_line(X, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
show_line(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
```

### 5.1.4 선형 모델 매개 변수의 해석해

경사 하강법은 반복 계산에 의해 근사값을 구하는 수치 계산법이다. __(수치해)__

직선 모델의 경우에는 방정식을 풀어서 정확한 해를 구할 수 있다. __(해석해)__

방정식을 풀어서 해석해를 구하면 1회의 계산만으로 최적의 w를 구할 수 있다.

해석해를 도출하면 문제의 본질을 잘 이해할 수 있고, 다차원 데이터에 대응하며, 곡선 모델로 확장하기 좋고, 커널법등의 이해를 도와준다.

<br>

목표 : 'J가 극소화되는 지점 w찾기'
그 지점의 기울기는 __'0'__ 이다.

해석해를 구해보자.
w0와 w1에 대해 연립방정식을 푼다.

```python
# 해석해 ------------------------------------
def fit_line(x, t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t * x)
    mxx = np.mean(x * x)
    w0 = (mtx - mt * mx) / (mxx - mx**2)
    w1 = mt - w0 * mx
    return np.array([w0, w1])

# 메인 ------------------------------------
W = fit_line(X, T)
print("w0={0:.3f}, w1={1:.3f}".format(W[0], W[1]))
mse = mse_line(X, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
plt.figure(figsize=(4, 4))
show_line(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
====
```

경사 하강법을 사용했을 때와 해석해를 구했을 때의 결과가 거의 같음을 볼 수 있다.
=> _직선으로 피팅한다면 해석해를 도출할 수 있으므로 경사 하강법을 사용할 필요가 없다._
그렇지만, 경사 하강법은 해석해를 사용할 수 없는 모델에서 유용하다.

<br>

<br>

## 5.2 2차원 입력면 모델
