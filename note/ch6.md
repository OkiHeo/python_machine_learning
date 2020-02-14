분류 문제(Classification)에서 목표 데이터는 __클래스__ .
ex) {0:과일, 1:야채, 2:곡물} 과 같이 정수를 할당할 수 있지만, 순서는 의미가 없는 카테고리.

<br>

## 6.1 1차원 입력 2클래스 분류

### 6.1.1 문제 설정
1차원 입력 변수 : xn
목표 변수 : tn  (0: 클래스0/ 1: 클래스1)

<br>

곤충 N마리의 데이터. 각각의 무게를 xn이라고 할 때 성별을 tn이라고 나타낸다.
무게를 통해 성별을 예측하는 모델을 만들어보자.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# 데이터 생성 --------------------------------
np.random.seed(seed=0) # 난수를 고정
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n) # 입력 데이터
T = np.zeros(X_n, dtype=np.uint8) # 목표 데이터
Dist_s = [0.4, 0.8] # 분포의 시작 지점
Dist_w = [0.8, 1.6] # 분포의 폭
Pi = 0.5 # 클래스 0의 비율 (암컷이 될 확률 0.5)
for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi) # (A)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]] # (B)
# 데이터 표시 --------------------------------
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))
====
X=[1.94 1.67 0.92 1.11 1.41 1.65 2.28 0.47 1.07 2.19 2.08 1.02 0.91 1.16
 1.46 1.02 0.85 0.89 1.79 1.89 0.75 0.9  1.87 0.5  0.69 1.5  0.96 0.53
 1.21 0.6 ]
T=[1 1 0 0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 1 0 1 0]
```

암컷이라면 Dist_s[0]=0.4 ~ Dist_w[0]=0.8에서 질량을 샘플링,
수컷이라면 Dist_s[1]=0.8 ~ Dist_w[1]=1.6에서 질량을 샘플링한다.

```python
# 데이터 분포 표시 ----------------------------
def show_data1(x, t):
    K = np.max(t) + 1
    for k in range(K): # (A)
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
                 linestyle='none', marker='o') # (B)
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(X_min, X_max)
        plt.yticks([0, 1])


# 메인 ------------------------------------
fig = plt.figure(figsize=(3, 3))
show_data1(X, T)
plt.show()
```

이미지@@@@@@@@@@@@@@@@@@@@@@@@@

`plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
         linestyle='none', marker='o') # (B)`는 처음 k의 처리는 t\==0일때 x와 t만을 추출하여 플롯하는 명령이다.
         t\==k를 만족하는 x[]와 t[]만을 추출할 수 있다.

수컷과 암컷을 분리하는 무게에 대한 경계선을 설정하면 문제를 해결할 수 있다.
__결정 경계(Decision boundary)__ 라고 한다.

선형 회귀 모델을 사용하면 질량이 충분히 커서 확실하게 수컷으로 판정할 수 있는 데이터 점에서도 직선이 데이터 점에 겹쳐져있지 않기 때문에 오차가 발생한다. -> 결정 경계가 수컷 쪽으로 끌려가서 제대로 피팅되지 않는다.

<br>

### 6.1.2 확률로 나타내는 클래스 분류

조건부 확률로 수컷일 확률을 알 수 있다.
무게가 0.8 < x < 1.2일 때 암컷 또는 수컷임을 확률 알아보면, 그 구간의 암컷의 수가 수컷보다 2배 더 많다면 수컷일 확률은 1/3이 된다.
-> x에 대해 t=1(수컷)일 확률은 P(t=1|x)로 나타낼 수 있다.

조건부 확률의 계단형 그래프는 클래스 분류의 답을 나타낸다고 생각할 수 있다. 어떤 클래스로 분류할지 명확하게 예측할 수 없는 불확실한 영역도 확률적인 예측으로 나타낸다. 이 방법은 불확실성을 명확하게 나타낼 수 있다는 점에서 직선에 의한 피팅보다 우수하다.

P(t=1|x) = 0.5가 되는 지점에 경계선을 그으면 된다.

### 6.1.3 최대가능도법

확률 w에서 t=1을 생성하는 모델 P(t=1|x)=w를 고려한다.
w는 0에서 1 사이.
T = 0, 0, 0, 1 이라는 데이터를 생성했다고 가정하면 w=1/4가 되지만, __다른 모델에 대해서도 대응할 수 있도록 최대가능도법을 사용__ 한다.

1. '모델에서 클래스 데이터 T=0, 0, 0, 1 이 생성될 확률'을 구한다. 이 확률을 가능도(우도)라고 한다.
2. P(t=0, 0, 0, 1|x) = (1-w)^3 * w
w : t=1이 되는 가능도.
이 그래프의 최댓값이 되는 지점의 w가 가장 적절한 값이 된다.

* 최대가능도법 : 주어진 입력 데이터 x에 대해 라벨 데이터 t가 생성될 확률(가능도)이 가장 커지는 w를 추정치로 한다.

필기@@@@@@@@@@@@@@@@@@@

<br>

### 6.1.4 로지스틱 회귀 모델

실제로 데이터가 계단식으로 균일하게 분포하는 경우는 거의 없다.
=> 로지스틱 회귀 확률로 나타낸다.

* 로지스틱 회귀 모델 : 직선의 식을 시그모이드 함수 안에 넣은 것.

y = w0x + w1를 시그모이드 함수 안에 넣으면
y = 1/(1+exp(-(w0x+w1))) 가 된다.
직선 모델의 큰 양의 출력은 1에 가까운 값이 되고, 절대값이 큰 음의 출력은 0에 가까운 값으로 변환된다.

```python
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    # 결정 경계
    i = np.min(np.where(y > 0.5)) # (A)
    B = (xb[i - 1] + xb[i]) / 2 # (B)
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    return B


# test
W = [8, -10]
show_logistic(W)
====

```

이미지@@@@@@@@@@@@@@@

<br>

### 6.1.5 교차 엔트로피 오차

로지스틱 회귀 모델을 통해 x가 t=1이 될 확률을 아래와 같이 나타낸다.
> y = σ(w0x+w1) = P(t=1|x)

일반적인 데이터에 대해 각 클래스의 생성 확률을
> P(t|x) = y^t * (1-y)^(1-t)

로 나타낸다.
-> P(t=0|x) = 1-y
-> P(t=1|x) = y

주어진 데이터가 N개라면 주어진 X에 대한 클래스 T의 생성확률은 아래와 같다.
하나하나의 데이터 생성 확률을 모든 데이터에 곱하면 된다.

필기@@@@@@@@@@@@@@@@

```python
# 평균 교차 엔트로피 오차 ---------------------
def cee_logistic(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n])) # -logP
    cee = cee / X_n   ## 1/N * -log P
    return cee


# test
W=[1,1]
cee_logistic(W, X, T)
```
평균 교차 엔트로피 오차를 그래프에서 확인해본다.

```python
from mpl_toolkits.mplot3d import Axes3D


# 계산 --------------------------------------
xn = 80 # 등고선 표시 해상도
w_range = np.array([[0, 15], [-15, 0]])
x0 = np.linspace(w_range[0, 0], w_range[0, 1], xn)
x1 = np.linspace(w_range[1, 0], w_range[1, 1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
C = np.zeros((len(x1), len(x0)))
w = np.zeros(2)
for i0 in range(xn):
    for i1 in range(xn):
        w[0] = x0[i0]
        w[1] = x1[i1]
        C[i1, i0] = cee_logistic(w, X, T)


# 표시 --------------------------------------
plt.figure(figsize=(12, 5))
#plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)
ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, C, color='blue', edgecolor='black',
                rstride=10, cstride=10, alpha=0.3)
ax.set_xlabel('$w_0$', fontsize=14)
ax.set_ylabel('$w_1$', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-15, 0)
ax.set_zlim(0, 8)
ax.view_init(30, -95)


plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, C, 20, colors='black',
                   levels=[0.26, 0.4, 0.8, 1.6, 3.2, 6.4])
cont.clabel(fmt='%1.1f', fontsize=8)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.grid(True)
plt.show()
```

이미지@@@@@@@@@@@@@@@@@

평균 교차 엔트로피 오차가 최소가 되는 값은 w0=9, w1=-9 근처에 존재할 것이다.

<br>

### 6.1.6 학습 규칙의 도출

교차 엔트로피 오차를 최소화하는 매개 변수의 분석해(==해석해)는 구할 수 없다. 왜냐하면 yn이 비선형의 시그모이드 함수를 포함하고 있기 때문이다.
-> 경사 하강법을 사용해서 수치적으로 구하는 것을 고려한다.

평균 교차 엔트로피 오차 E(w)를 w0, w1로 편미분하여 =0이 되는 지점을 찾자.

@@@@@@필기

프로그래밍으로 구현하면 아래와 같다.

```python
# 평균 교차 엔트로피 오차의 미분 --------------
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X_n
    return dcee


# --- test
W=[1, 1]
dcee_logistic(W, X, T)
====
array([0.30857905, 0.39485474]) ## w0방향, w1방향의 편미분 값.
```

<br>

### 6.1.7 경사 하강법에 의한 해

`scipy.optimize` 라이브러리에 포함된 `minimize()` 함수로 경사 하강법을 시도한다.

```python
from scipy.optimize import minimize


# 매개 변수 검색
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x, t),
                    jac=dcee_logistic, method="CG") # (A)
    return res1.x


# 메인 ------------------------------------
plt.figure(1, figsize=(3, 3))
W_init=[1,-1]
W = fit_logistic(W_init, X, T)
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
B=show_logistic(W)
show_data1(X, T)
plt.ylim(-.5, 1.5)
plt.xlim(X_min, X_max)
cee = cee_logistic(W, X, T)
print("CEE = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(B))
plt.show()
```

@@@@@@@@@@@@@그래프

`minimize()`의 인수로, 교차 엔트로피 함수 cee_logistic, w의 초기값 w_init, args=(x, t)에는 cee_logistic의 w이외의 인수. jac=dcee_logistic에는 미분 함수를 지정. method로 CG(켤레 기울기법)지정.

<br>

## 6.2 2차원 입력 2클래스 분류

### 6.2.1 문제 설정

데이터의 수 N=100,
입력 데이터는 Nx2의 X에 저장,
2클래스 분류의 클래스 데이터는 Nx2의 T2에,
3클래스 분류의 클래스 데이터는 Nx2의 T3에 저장한다.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# 데이터 생성 --------------------------------
np.random.seed(seed=1)  # 난수를 고정
N = 100 # 데이터의 수
K = 3 # 분포 수
T3 = np.zeros((N, 3), dtype=np.uint8)
T2 = np.zeros((N, 2), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0 범위 표시 용
X_range1 = [-3, 3] # X1의 범위 표시 용
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # (A) 각 분포에 대한 비율 0.4 0.8 1
for n in range(N):
    wk = np.random.rand()
    for k in range(K): # (B) wk에 난수저장하고 클래스분류
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k]
                   + Mu[T3[n, :] == 1, k])
T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]
```

```python
print(X[:5,:])
====
[[-0.14173827  0.86533666]
 [-0.86972023 -1.25107804]
 [-2.15442802  0.29474174]
 [ 0.75523128  0.92518889]
 [-1.10193462  0.74082534]]
```

```python
print(T2[:5,:])
====
[[0 1]
 [1 0]
 [1 0]
 [0 1]
 [1 0]]
```
값이 1인 열 번호가 클래스 번호를 나타낸다.
1, 0, 0 ,1, 0클래스에 속한다는 의미.

```python
print(T3[:5,:])
====
[[0 1 0]
 [1 0 0]
 [1 0 0]
 [0 1 0]
 [1 0 0]]
```
클래스 1, 0, 0, 1, 0에 속한다는 의미.

이처럼 목적 변수 벡터 tn의 k번째 요소만 1으로, 그 외에는 0으로 표기하는 방법을 __1-of-K 부호화__ 라고 한다.

T2와 T3를 그림으로 나타내면 아래와 같다.
```python
# 데이터 표시 --------------------------
def show_data2(x, t):
    wk, K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=c[k], alpha=0.8)
        plt.grid(True)


# 메인 ------------------------------
plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
show_data2(X, T2)
plt.xlim(X_range0)
plt.ylim(X_range1)


plt.subplot(1, 2, 2)
show_data2(X, T3)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()
```

@@@@@@@@@@@@@@@이미지

__\# (A) 각 분포에 대한 비율 0.4 0.8 1__
어떤 클래스에 속하는 확률을 `Pi=np.array([0.4, 0.8, 1])`으로 설정하고 ,

__\# (B) wk에 난수저장하고 클래스분류__
0~1 사이의 균일한 분포에서 난수를 생성하여 wk에 넣었다.
wk의 각 값이 Pi[0]보다 작으면 클래스 0, Pi[1]보다 작으면 클래스 1, Pi[2]보다 작으면 클래스 2로 한다.

클래스가 결정된 뒤, 클래스마다 각각 다른 가우스 분포로 입력 데이터를 생성한다.

<br>

### 6.2.2 로지스틱 회귀 모델

모델의 매개 변수를 하나 증가시켜
y = σ(a)
a = w0x0 + w1x1 + w2
가 된다.

이번 모델의 출력 y는 P(t=0|x)를 근사하는 것으로 한다.

모델의 정의
```python
# 로지스틱 회귀 모델 -----------------
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y
```
모델과 데이터를 3D로 표시한다.
```python
# 모델 3D보기 ------------------------------
from mpl_toolkits.mplot3d import axes3d


def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)


def show_data2_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    Ax.view_init(elev=25, azim=-30)


# test ---
Ax = plt.subplot(1, 1, 1, projection='3d')
W=[-1, -1, -1]
show3d_logistic2(Ax, W)
show_data2_3d(Ax,X,T2)
```

@@@@@@@@이미지

W = [-1, -1, -1]을 선택한 경우의 로지스틱 회귀 모델의 모형.

```python
# 모델 등고선 2D 표시 ------------------------


def show_contour_logistic2(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)


# test ---
plt.figure(figsize=(3,3))
W=[-1, -1, -1]
show_contour_logistic2(W)
```

@@@@@@@@@이미지

모델의 평균 교차 엔트로피 함수는 이전의 E(w)와 동일하다.

데이터베이스에는 1-of-K 부호화를 사용하는데, 2클래스 분류 문제이므로 T의 0열 t_n0을 tn으로 두고 1이면 클래스0, 0이면 클래스 1으로 처리한다.


```python
# 크로스 엔트로피 오차 ------------
def cee_logistic2(w, x, t):
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) +
                     (1 - t[n, 0]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee
```

매개변수의 편미분으로 E(w)의 기울기가 0이 되는 지점의 w0, w1, w2를 구한다.
@@@@@@@필기

```python
# 크로스 엔트로피 오차의 미분 ------------
def dcee_logistic2(w, x, t):
    X_n=x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / X_n
    return dcee


# test ---
W=[-1, -1, -1]
dcee_logistic2(W, X, T2)
====
array([ 0.10272008,  0.04450983, -0.06307245])
```

평균 교차 엔트로피 오차가 최소가 되도록 로지스틱 회귀 모델의 매개 변수를 구하고 결과를 표시한다.

```python
from scipy.optimize import minimize


# 로지스틱 회귀 모델의 매개 변수 검색 -
def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x, t),
                   jac=dcee_logistic2, method="CG")
    return res.x


# 메인 ------------------------------------
plt.figure(1, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)


Ax = plt.subplot(1, 2, 1, projection='3d')
W_init = [-1, 0, 0]
W = fit_logistic2(W_init, X, T2)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
show3d_logistic2(Ax, W)


show_data2_3d(Ax, X, T2)
cee = cee_logistic2(W, X, T2)
print("CEE = {0:.2f}".format(cee))


Ax = plt.subplot(1, 2, 2)
show_data2(X, T2)
show_contour_logistic2(W)
plt.show()
```

@@@@@@@@@@이미지

평면을 넣었으므로, 이 모델의 결정 경계(푸른색)는 직선이 된다.

<br>

<br>

## 6.3 2차원 입력 3클래스 분류

### 6.3.1 3클래스 분류 로지스틱 회귀 모델

소프트맥스 함수를 모델의 출력에 사용하는 것으로 3클래스 이상의 클래스 분류에 대응 가능하다.

@@@@@@@@@ 필기

w_ki는 입력 xi에서 클래스 k의 입력 총합을 조절하는 매개변수.

3클래스용 로지스틱 회귀 모델 `logistic3`를 구현한다.

```python
# 3 클래스 용 로지스틱 회귀 모델 -----------------

def logistic3(x0, x1, w):
    K = 3
    w = w.reshape((3, 3))
    n = len(x1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y


# test ---
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
y = logistic3(X[:3, 0], X[:3, 1], W)
print(np.round(y, 3))
====
[[0.    0.006 0.994]
 [0.965 0.033 0.001]
 [0.925 0.07  0.005]]
```

이 모델의 매개 변수 w는 9개이다.
`minimize`에 대응하기 위해 W는 3x3행렬을 늘어놓은 요소 수 9개의 벡터로 취급한다.

test에서는 위부터 3개의 입력 데이터 X[:3, 0]와 시험적으로 결정한 W에 대한 출력 y를 확인하고 있다.

출력은 Nx3의 행렬로 표현한 y로, 같은 라인의 요소를 더하면 1이 된다.

<br>

### 6.3.2 교차 엔트로피 오차
가능도는 모든 입력 데이터 X에 대해서 전체 클래스 데이터 T가 생성된 확률이다.

3클래스에서는 아래와 같이 표현할 수 있다.

@@@@@@@@@@@@ 필기

교차 엔트로피 오차를 계산하는 함수 `cee_logistic3`를 정의하자.

```python
# 교차 엔트로피 오차 ------------
def cee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
cee_logistic3(W, X, T3)
====
3.9824582404787288
```
9개 요소의 배열 W와, X, T3를 인수로 스칼라 값을 출력한다.

<br>

### 6.3.3 경사 하강법에 의한 해

E(w)의 각 w_ki에 관한 편미분이 필요하다.
아래와 같이 구할 수 있다.

@@@@@@@@ 필기

이는 모든 k와 i에 대해 동일한 형태를 가진다.

각 매개 변수에 대한 미분값을 출력하는 함수 `dcee_logistic3`를 정의한다.

```python
# 교차 엔트로피 오차의 미분 ------------
def dcee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    dcee = np.zeros((3, 3)) # (클래스의 수 K) x (x의 차원 D+1)
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])* np.r_[x[n, :], 1]
    dcee = dcee / X_n
    return dcee.reshape(-1)


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
dcee_logistic3(W, X, T3)
====
array([ 0.03778433,  0.03708109, -0.1841851 , -0.21235188, -0.44408101,
       -0.38340835,  0.17456754,  0.40699992,  0.56759346])
```

출력은 각 편미분에 대응한 요소 수 9개의 배열이 된다.
이를 `minimize`에 전달하여 매개 변수 검색을 수행하는 함수를 만든다.

```python
# 매개 변수 검색 -----------------
def fit_logistic3(w_init, x, t):
    res = minimize(cee_logistic3, w_init, args=(x, t),
                   jac=dcee_logistic3, method="CG")
    return res.x
```

등고선에 결과를 표시하는 함수 `show_contour_logistic3`도 만들어둔다.
가중치 매개 변수 w를 전달하면, 표시할 입력 공간을 30x30으로 분할하여 모든 입력에 대해 네트워크의 출력을 확인한다.

```python
# 모델 등고선 2D 표시 --------------------
def show_contour_logistic3(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)


    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic3(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j],
                           levels=(0.5, 0.9),
                           colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)
```

```python
# 메인 ------------------------------------
W_init = np.zeros((3, 3))
W = fit_logistic3(W_init, X, T3)
print(np.round(W.reshape((3, 3)),2))
cee = cee_logistic3(W, X, T3)
print("CEE = {0:.2f}".format(cee))


plt.figure(figsize=(3, 3))
show_data2(X, T3)
show_contour_logistic3(W)
plt.show()
```

@@@@@@@@@@이미지

클래스 사이에 경계선이 그어져있다.
클래스 간 경계선이 직선의 조합으로 구성된다.

<br>

이 모델의 훌륭한 점은 __모호성을 조건부 확률(사후 확률)로 근사하는 것에 있다.__
