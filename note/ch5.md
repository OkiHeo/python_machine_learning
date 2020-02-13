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

1차원의 경우에는 나이(x)만을 이용해서 키를 예측했지만,
2차원 모델에서는 나이(x0)와 몸무게(x1)를 사용해서 키를 예측한다.

데이터에 포함되는 사람의 체질량 지수가 평균 23이라고 가정해서 키를 이용해 몸무게 데이터를 만든다.
> 몸무게 = 23*(키^2)/100 + 노이즈

```python
# 2차원 데이터 생성 --------------------------
X0 = X
X0_min = 5
X0_max = 30
np.random.seed(seed=1) # 난수를 고정
X1 = 23 * (T / 100)**2 + 2 * np.random.randn(X_n)
X1_min = 40
X1_max = 75

print(np.round(X0, 2))    #나이
print(np.round(X1, 2))    #몸무게
print(np.round(T, 2))     #키
====
[15.43 23.01  5.   12.56  8.67  7.31  9.66 13.64 14.92 18.47 15.48 22.13
 10.11 26.95  5.68 21.76]
[70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95
 57.73 66.89 46.68 61.08]
[170.91 160.68 129.   159.7  155.46 140.56 153.65 159.43 164.7  169.65
 160.71 173.29 159.31 171.52 138.96 165.87]
```
16명의 나이, 몸무게, 키 데이터를 그래프로 나타내면 아래와 같다.
```python
# 2차원 데이터의 표시 ------------------------
def show_data2(ax, x0, x1, t):
    for i in range(len(x0)):
        ax.plot([x0[i], x0[i]], [x1[i], x1[i]],
                [120, t[i]], color='gray')
        ax.plot(x0, x1, t, 'o',
                color='cornflowerblue', markeredgecolor='black',
                markersize=6, markeredgewidth=0.5)
        ax.view_init(elev=35, azim=-75)


# 메인 ------------------------------------
plt.figure(figsize=(6, 5))
ax = plt.subplot(1,1,1,projection='3d')
show_data2(ax, X0, X1, T)
plt.show()
====

```
이미지@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

### 5.2.2 면 모델
N개의 2차원 벡터 x_n에 대해 각각 t_n이 할당되어 있으므로 이 관계를 그래프로 보면 아래와 같다.

__`show_plane(ax, w)`__ : 임의의 w에 대해 면을 그리는 함수
__aw__ : 3차원 그래프를 그릴 때 필요한 묘사 대상 그래프의 id.
ex) 아래의 예시에서는 위에서 사용한 나이,몸무게,키를 나타낸 그래프 ax를 사용했다.

```python
#면의 표시 ----------------------------------
def show_plane(ax, w):
    px0 = np.linspace(X0_min, X0_max, 5)
    px1 = np.linspace(X1_min, X1_max, 5)
    px0, px1 = np.meshgrid(px0, px1)
    y = w[0]*px0 + w[1] * px1 + w[2]
    ax.plot_surface(px0, px1, y, rstride=1, cstride=1, alpha=0.3,
                    color='blue', edgecolor='black')

#면의 MSE -----------------------------------
def mse_plane(x0, x1, t, w):
    y = w[0] * x0 + w[1] * x1 + w[2] # (A)
    mse = np.mean((y - t)**2)
    return mse

# 메인 ------------------------------------
plt.figure(figsize=(6, 5))
ax = plt.subplot(1, 1, 1, projection='3d')
W = [1.5, 1, 90]
show_plane(ax, W)
show_data2(ax, X0, X1, T)
mse = mse_plane(X0, X1, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
plt.show()
====
```
이미지@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

y(x) = w0*x0 + w1*x1 + w2
w0, w1, w2에 다양한 값을 넣어서 여러 위치, 기울기를 가진 면을 만들 수 있다.

### 5.2.3 매개 변수의 해석해

2차원 면 모델에서도 1차원 선 모델과 마찬가지로 평균 제곱 오차를 정의할 수 있다.
y_n이 면을 나타내는 식이라는 것!

w를 움직이면 면이 여러 방향을 향하며, 그에 따라 J가 변화한다.
J를 최소화하는 w0, w1, w2를 찾는 것이 우리의 목표.
J를 각각 w0, w1, w2로 편미분하여 =0이 되는 지점을 찾으면 그 지점이 J를 최소화하는 지점이 된다.


해석해로 구한 w0, w1, w2를 실제로 구하고, 그래프를 그려보면 아래와 같다.

```python
# 해석해 ------------------------------------
def fit_plane(x0, x1, t):
    c_tx0 = np.mean(t * x0) - np.mean(t) * np.mean(x0)
    c_tx1 = np.mean(t * x1) - np.mean(t) * np.mean(x1)
    c_x0x1 = np.mean(x0 * x1) - np.mean(x0) * np.mean(x1)
    v_x0 = np.var(x0)
    v_x1 = np.var(x1)
    w0 = (c_tx1 * c_x0x1 - v_x1 * c_tx0) / (c_x0x1**2 - v_x0 * v_x1)
    w1 = (c_tx0 * c_x0x1 - v_x0 * c_tx1) / (c_x0x1**2 - v_x0 * v_x1)
    w2 = -w0 * np.mean(x0) - w1 * np.mean(x1) + np.mean(t)
    return np.array([w0, w1, w2])


# 메인 ------------------------------------
plt.figure(figsize=(6, 5))
ax = plt.subplot(1, 1, 1, projection='3d')
W = fit_plane(X0, X1, T)
print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
show_plane(ax, W)
show_data2(ax, X0, X1, T)
mse = mse_plane(X0, X1, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
plt.show()
====

```
이미지@@@@@@@@@@@@@@@@@@@@@@@@@@@@

<br>

## 5.3 D차원 선형 회귀 모델

### 5.3.1 D차원 선형 회귀 모델

### 5.3.2 매개변수의 해석해

### 5.3.3 원점을 지나지 않는 면에 대한 확장

<br>

<br>

## 5.4 선형 기저 함수 모델

곡선을 사용해서 오차함수를 계산하면 오차가 더 작아질 수 있다.

기저 함수 : 바탕이 되는 함수.
가우스 함수를 기저 함수로 선택한 선형 기저 함수 모델을 생각하자.
기저 함수는 ∮_$_ j$(x)로 나타낸다.
기저 함수는 여러 세트에서 사용되기 떄문에 그 번호를 나타나는 j에는 인덱스가 붙어있다.

가우스 함수의 중심 위치는 μ_$_ j$이다.
함수의 확장 정도는 s로 조절한다. s는 모든 가우스 함수에 공통 매개변수가 된다.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# 데이터 로드 ----------------------------
outfile = np.load('ch5_data.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_n = outfile['X_n']
T = outfile['T']

# 가우스 함수 ---------------------------------
def gauss(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

# 메인 ------------------------------------
M = 4   # 가우스 함수 4개.
plt.figure(figsize=(4, 4))
mu = np.linspace(5, 30, M)  #4개의 가우스함수의 중심 설정.
s = mu[1] - mu[0] # (A)
xb = np.linspace(X_min, X_max, 100)
for j in range(M):
    y = gauss(xb, mu[j], s)
    plt.plot(xb, y, color='gray', linewidth=3)
plt.grid(True)
plt.xlim(X_min, X_max)
plt.ylim(0, 1.2)
plt.show()
====

```

<br>

```python
# 선형 기저 함수 모델 ----------------
def gauss_func(w, x):
    m = len(w) - 1
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    y = np.zeros_like(x) # x와 같은 크기로 요소가 0의 행렬 y를 작성
    for j in range(m):
        y = y + w[j] * gauss(x, mu[j], s)
    y = y + w[m]
    return y

# 선형 기저 함수 모델 MSE ----------------
def mse_gauss_func(x, t, w):
    y = gauss_func(w, x)
    mse = np.mean((y - t)**2)
    return mse

# 선형 기저 함수 모델 정확한 솔루션 -----------------
def fit_gauss_func(x, t, m):
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    n = x.shape[0]
    psi = np.ones((n, m+1))
    for j in range(m):
        psi[:, j] = gauss(x, mu[j], s)
    psi_T = np.transpose(psi)


    b = np.linalg.inv(psi_T.dot(psi))
    c = b.dot(psi_T)
    w = c.dot(t)
    return w
```
```python
# 가우스 기저 함수 표시 -----------------------
def show_gauss_func(w):
    xb = np.linspace(X_min, X_max, 100)
    y = gauss_func(w, xb)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)


# 메인 ----------------------------------
plt.figure(figsize=(4, 4))
M = 4
W = fit_gauss_func(X, T, M)
show_gauss_func(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
mse = mse_gauss_func(X, T, W)
print('W='+ str(np.round(W,1)))
print("SD={0:.2f} cm".format(np.sqrt(mse)))
plt.show()
```


<br>

<br>

## 5.5. 오버피팅의 문제

M을 변화시킴에 따라 달라지는 선형 기저함수 피팅.
```python
plt.figure(figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.3)
M = [2, 4, 7, 9]
for i in range(len(M)):
    plt.subplot(1, len(M), i + 1)
    W = fit_gauss_func(X, T, M[i])
    show_gauss_func(W)
    plt.plot(X, T, marker='o', linestyle='None',
             color='cornflowerblue', markeredgecolor='black')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.ylim(130, 180)
    mse = mse_gauss_func(X, T, W)


    plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()
```

이미지@@@@@@@@@@@@@@

M이 커지면 오차가 감소하지만, 가우스 기저 함수가 흐물흐물해져서 새롭게 들어오는 데이터에 대해 쉽게 예측할 수 없다.
이러한 현상을 __과적합__ 이라고 한다.

M=2~9에 따른 오차를 그래프로 나타내서 보면 아래와 같다.

```python
plt.figure(figsize=(5, 4))
M = range(2, 10)
mse2 = np.zeros(len(M))
for i in range(len(M)):
    W = fit_gauss_func(X, T, M[i])
    mse2[i] = np.sqrt(mse_gauss_func(X, T, W))
plt.plot(M, mse2, marker='o',
         color='cornflowerblue', markeredgecolor='black')
plt.grid(True)
plt.show()
```

이미지@@@@@@@@@@

M이 증가함에 따라 오차(SD)가 점점 감소한다.
데이터 점이 없는 곳은 MSE와 관계없어서 데이터 점이 없는 곳에서는 곡선이 뒤틀려버린다.

_데이터 점의 오차는 작아져도, 새 데이터의 예측은 나빠지는 과적합(오버피팅)현상이 나타난다._

<br>

새 데이터에 대한 예측의 정확도를 높이려면
데이터 X와 t의 1/4를 테스트 데이터로, 나머지 3/4를 훈련 데이터로 나눈다.
모델 매개 변수 __w는 훈련 데이터만을 사용해서 최적화한다.__

* 홀드 아웃(Holdout) 검증
w를 사용해서 테스트 데이터의 평균 제곱 오차를 계산하고, M의 평가 기준으로 사용한다.
=> 훈련에 이용하지 않은 미지의 데이터에 대한 예측 오차로 M을 평가한다.

```python
# 훈련 데이터와 테스트 데이터 ------------------
X_test = X[:int(X_n / 4 + 1)]
T_test = T[:int(X_n / 4 + 1)]
X_train = X[int(X_n / 4 + 1):]
T_train = T[int(X_n / 4 + 1):]
# 메인 ------------------------------------
plt.figure(figsize=(10, 2.5))


plt.subplots_adjust(wspace=0.3)
M = [2, 4, 7, 9]
for i in range(len(M)):
    plt.subplot(1, len(M), i + 1)
    W = fit_gauss_func(X_train, T_train, M[i])
    show_gauss_func(W)
    plt.plot(X_train, T_train, marker='o',
             linestyle='None', color='white',
             markeredgecolor='black', label='training')
    plt.plot(X_test, T_test, marker='o', linestyle='None',
             color='cornflowerblue',
             markeredgecolor='black', label='test')
    plt.legend(loc='lower right', fontsize=10, numpoints=1)
    plt.xlim(X_min, X_max)
    plt.ylim(130, 180)
    plt.grid(True)
    mse = mse_gauss_func(X_test, T_test, W)
    plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()
```

이미지@@@@@@@@@@@@@@@@

M을 2부터 9까지 이동해 가면서 훈련 데이터와 테스트 데이터의 오차(SD)를 그래프에 나타낸다면 아래와 같다.

```python
plt.figure(figsize=(5, 4))
M = range(2, 10)
mse_train = np.zeros(len(M))
mse_test = np.zeros(len(M))
for i in range(len(M)):
    W = fit_gauss_func(X_train, T_train, M[i])
    mse_train[i] = np.sqrt(mse_gauss_func(X_train, T_train, W))
    mse_test[i] = np.sqrt(mse_gauss_func(X_test, T_test, W))
plt.plot(M, mse_train, marker='o', linestyle='-',
         markerfacecolor='white', markeredgecolor='black',
         color='black', label='training')
plt.plot(M, mse_test, marker='o', linestyle='-',
         color='cornflowerblue', markeredgecolor='black',
         label='test')
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, 12)
plt.grid(True)
plt.show()
```

이미지@@@@@@@@@@@@@@@@@@@@@@

M이 4에서 5로 넘어가면서 SD가 증가하는 경향으로 변한다.
-> M=5에서 과적합이 일어나고 있다.

<br>

데이터의 분류를 다르게 함에 따라 홀드 아웃의 결과가 달라진다.

데이터의 분류에 따른 홀드아웃의 차이를 최대한 줄이는 교차 검증 방법을 사용한다.
K겹 교차 검증으로 부르기도 한다.

데이터 X와 t를 K분할하여 1번째를 테스트 데이터, 나머지를 훈련 데이터로, 2번째를 테스트 데이터 나머지를 훈련 데이터...
마지막에 K개의 평균 제곱 오차의 평균을 계산하여 이 값을 M의 평가값으로 한다.

데이터를 K분할하여 각각의 평균 제곱 오차를 출력하는 함수 `kfold_gauss_func(x, t, m, k)`.

```python
# K 분할 교차 검증 -----------------------------
def kfold_gauss_func(x, t, m, k):
    n = x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)
    for i in range(0, k):
        x_train = x[np.fmod(range(n), k) != i] # (A)
        t_train = t[np.fmod(range(n), k) != i] # (A)
        x_test = x[np.fmod(range(n), k) == i] # (A)
        t_test = t[np.fmod(range(n), k) == i] # (A)
        wm = fit_gauss_func(x_train, t_train, m)
        mse_train[i] = mse_gauss_func(x_train, t_train, wm)
        mse_test[i] = mse_gauss_func(x_test, t_test, wm)
    return mse_train, mse_test
```
`np.fmod(n, k)`는 n을 k로 나눈 나머지를 출력한다.
```python
np.fmod(range(10),5)
```

```python
M = 4
K = 4
kfold_gauss_func(X, T, M, K)
====
(array([12.87927851,  9.81768697, 17.2615696 , 12.92270498]),
 array([ 39.65348229, 734.70782012,  18.30921743,  47.52459642]))
```

`khold_gauss_func`함수로 분할 수를 최대의 16으로 하고, 2에서 7까지의 M으로 오차의 평균을 계산해서 그래프로 나타낸다.

```python
M = range(2, 8)
K = 16
Cv_Gauss_train = np.zeros((K, len(M)))
Cv_Gauss_test = np.zeros((K, len(M)))
for i in range(0, len(M)):
    Cv_Gauss_train[:, i], Cv_Gauss_test[:, i] =\
                    kfold_gauss_func(X, T, M[i], K)
mean_Gauss_train = np.sqrt(np.mean(Cv_Gauss_train, axis=0))
mean_Gauss_test = np.sqrt(np.mean(Cv_Gauss_test, axis=0))


plt.figure(figsize=(4, 3))
plt.plot(M, mean_Gauss_train, marker='o', linestyle='-',
         color='k', markerfacecolor='w', label='training')
plt.plot(M, mean_Gauss_test, marker='o', linestyle='-',
         color='cornflowerblue', markeredgecolor='black', label='test')
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, 20)
plt.grid(True)
plt.show()
```

이미지@@@@@@@@@@@@@@@@@@@@@@@@@

__교차 검증은 M을 구하기 위한 방법이며, 모델 매개 변수 w를 구하는 용도가 아니다.__
이제 M=3이 최적임을 알고 있으므로, 그 모델의 매개 변수 w를 모든 데이터를 사용해 마지막으로 계산한다.

```python
M = 3
plt.figure(figsize=(4, 4))
W = fit_gauss_func(X, T, M)
show_gauss_func(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.xlim([X_min, X_max])
plt.grid(True)
mse = mse_gauss_func(X, T, W)
print("SD={0:.2f} cm".format(np.sqrt(mse)))
plt.show()
```

이미지@@@@@@@@@@@@@@@@@@@@

이번 테스트 데이터(N=16)처럼 데이터의 수가 적은 경우에는 교차 검증이 유용하다.
하지만, 데이터 수가 많으면 교차 검증은 계산시간이 너무 오래 걸리므로 홀드 아웃 검증을 사용하면 좋다. 데이터 수가 크면 홀드 아웃 검증과 교차 검증의 결과가 비슷해진다.

<br>

<br>

## 5.6 새로운 모델의 생성

위의 그래프를 보았을 때 25세쯤에서 키가 줄어드는 현상을 볼 수 있다.
30세 근처의 데이터가 존재하지 않기 때문에 이런 현상이 나타난다.

'키는 나이가 들면서 점차 커지고, 일정한 곳에서 수렴한다'는 지식을 모델에 추가하기 위해서는 _그 지식에 해당하는 모델을 만들면_ 된다.

식@@@@@@@@@@@@@@

x가 증가하면 exp(-w2*x)는 0에 접근한다.
-> y(x)가 w0에 수렴하게 된다.

평균 제곱 오차 J가 최소가 되도록 w0, w1, w2를 선택한다.

식@@@@@@@@@@@@@@@

> 함수의 최솟값, 최댓값을 구하는 문제는 '최적화 문제'라고 하는데, 이러한 문제를 푸는 라이브러리로 scipy.optimize에 포함된 minimize함수를 사용한다.

모델 A를 `model_A(x, w)`로 정의하고, 표시용 함수 `show_model_A(w)`와 MSE를 출력하는 함수 `mse_model_A(w, x, t)`를 정의한다.

```python
# 모델 A -----------------------------------
def model_A(x, w):
    y = w[0] - w[1] * np.exp(-w[2] * x)
    return y


# 모델 A 표시 -------------------------------
def show_model_A(w):
    xb = np.linspace(X_min, X_max, 100)
    y = model_A(xb, w)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)


# 모델 A의 MSE ------------------------------
def mse_model_A(w, x, t):
    y = model_A(x, w)
    mse = np.mean((y - t)**2)
    return mse
```

매개 변수의 최적화...

```python
from scipy.optimize import minimize


# 모델 A의 매개 변수 최적화 -----------------
def fit_model_A(w_init, x, t):
    res1 = minimize(mse_model_A, w_init, args=(x, t), method="powell")
    return res1.x
```
`fit_model_A(w_init, x, t)`의 w_init은 반복 연산을 위한 매개 변수의 초기값, 입력 데이터 x, 목표 데이터 t.
`minimize`함수
첫 번째 인수 : 최소화할 목표 함수
두 번째 인수 : w의 초기값
세 번째 인수 : 목표 함수를 최적화하는 매개 변수 w이외의 인수
+) 옵션으로 method를 "powell"로 지정하여 구배를 사용하지 않는 최적화 방법인 파웰 알고리즘을 지정한다.

```python
# 메인 ------------------------------------
plt.figure(figsize=(4, 4))
W_init=[100, 0, 0]
W = fit_model_A(W_init, X, T)
print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
show_model_A(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue',markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
mse = mse_model_A(W, X, T)
print("SD={0:.2f} cm".format(np.sqrt(mse)))
plt.show()
```

이미지@@@@@@@@@@@@@@

직선 모델보다 오차가 적으며, M=3의 선형 기저 함수 모델에 비해서도 낮은 값을 가진다.
그래프는 나이가 들면서 동시에 키가 커지며, 일정한 값에서 수렴한다는 형태를 보이고 있다.

<br>

## 5.7 모델의 선택

어떤 모델이 더 좋을지, 모델 간의 비교가 필요하다.

모델 간의 비교에도 선형 기저 함수의 모델 M을 결정할 때와 동일한 아이디어,
**미지의 데이터에 대한 예측 정확도로 평가한다** 가 유효하다.
=> 홀드 아웃 검증, 교차 검증 모델로 모델의 좋고 나쁨을 평가할 수 있다.

<br>

모델 A의 LOOCV를 실시하여, 선형 기저 함수 모델의 결과와 비교한다.

```python
# 교차 검증 model_A ---------------------------
def kfold_model_A(x, t, k):
    n = len(x)
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)
    for i in range(0, k):
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = fit_model_A(np.array([169, 113, 0.2]), x_train, t_train)
        mse_train[i] = mse_model_A(wm, x_train, t_train)
        mse_test[i] = mse_model_A(wm, x_test, t_test)
    return mse_train, mse_test


# 메인 ------------------------------------
K = 16
Cv_A_train, Cv_A_test = kfold_model_A(X, T, K)
mean_A_test = np.sqrt(np.mean(Cv_A_test))
print("Gauss(M=3) SD={0:.2f} cm".format(mean_Gauss_test[1]))
print("Model A SD={0:.2f} cm".format(mean_A_test))
SD = np.append(mean_Gauss_test[0:5], mean_A_test)
M = range(6)
label = ["M=2", "M=3", "M=4", "M=5", "M=6", "Model A"]
plt.figure(figsize=(5, 3))
plt.bar(M, SD, tick_label=label, align="center",
facecolor="cornflowerblue")
plt.show()
```

이미지@@@@@@@@@@@@@

선형 기저 함수 모델의 그래프와 비교했을 때, 모델 A의 SD가 더 작다.
-> 선형 기저 함수 모델보다 모델 A가 데이터에 잘 어울린다.

<br>

<br>

## 5.8 정리

지도 학습의 회귀 문제의 전체 흐름

1. 입력 변수와 목표 변수의 데이터가 있다.
2. 무엇을 가지고 예측의 정확도를 높일지, __목적 함수를 결정__ 한다. ex)평균 제곱 오차 함수
3. 모델의 후보를 생각한다.
    선형 회귀 모델/곡선 모델/...
4. 데이터를 테스트 데이터와 훈련 데이터로 나눈다.
5. 훈련 데이터를 사용하여 원하는 함수가 최소(/최대)가 되도록 각 모델의 매개 변수 w를 결정한다.
6. 이 모델 매개 변수를 사용하여 테스트 데이터의 입력 데이터 X에서 목표 데이터 t의 예측을 실시하여 가장 오차가 적은 모델을 선택한다.
7. 모델이 결정되면 보유한 데이터를 모두 사용하여 모델 매개 변수를 최적화한다.
