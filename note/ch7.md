* 딥러닝 : 머신러닝(기계학습)의 한 방법. 신경망 모델이라고 불리는 뇌의 신경 네트워크에서 힌트를 얻은 알고리즘. 그 중에서도 층을 많이 활용한 모델이 '딥러닝'이다.

* 심층 신경망(DNN, Deep Neural Network)이라고도 함.
  -  '깊은 층으로 이루어진 신경망 모델.'

<br>

## 7.1 뉴런 모델
신경망 모델은 '뉴런 모델' 단위로 구축된다.

### 7.1.2 뉴런 모델
신경 세포의 움직임을 단순화한 수학적 모델인 뉴런 모델.

뉴런에 두 개의 입력 __x=(x0, x1)__ 이 들어오는 것을 가정한다.

각각의 입력에 대한 __시냅스 전달 강도를 w0, w1__ 로 하여 이들을 곱하여 모든 입력으로 합을 얻은 상수 w2를 더한 것을 __입력 총합 a__ 로 한다.

> a = w0x0 + w1x1 + w2

모든 입력에 대해 항상 1의 값을 갖는 더미 입력 x2를 가정하여

> a = w0x0 + w1x1 + w2x2

> a = $\sum_{i=0}^2 {w_ix_i}$

입력 총합 a를 시그모이드 함수에 적용시킨 식을 __뉴런의 출력값 y__ 로 한다.

> y = 1/ {1+ exp(-a)}

시그모이드 함수에 따라 y는 0~1 사이의 연속된 값을 가진다.

a가 클수록 y가 1에 가까워져 펄스를 많이 보내고, a가 작을수록 y가 0에 가까워져 펄스를 거의 보내지 않는다.

=> 이 모델은 6.1.4에서 설명한 '로지스틱 회귀 모델'이 된다. 2차원 공간을 직선으로 나누어 한 쪽은 0~0.5, 다른 한 쪽은 0.5~1의 숫자를 할당하는 기능을 가진다.

<br>

입력 공간에 대한 입력 총합은 평면으로 표시된다. 입력 총합이 0이 되는 것은 직선 a=0 위의 입력이 된다.

<br>

입력 차수에 2 대신 'D'를 적용한 일반적인 경우에 식은 아래와 같다.

@@@@@@ 필기

* 뉴런 모델은 D차원의 공간을 D-1차원 평면에서 2개로 나눈다고 할 수 있다.

* N개의 데이터셋 (xn, tn)에 대한 뉴런 모델의 학습 방법

1. 목적 함수 : 평균 교차 엔트로피 오차
  @@@@@@@@@ 필기
2. 오차 함수의 매개 변수에 대한 기울기
  @@@@@@@@@ 필기
3. 매개 변수의 학습 법칙은 그 기울기를 사용하여 아래와 같다.
  @@@@@@@@@ 필기

<br>

<br>

## 7.2 신경망 모델

### 7.2.1 2층 피드 포워드 신경망

뉴런의 집합체 모델을 신경망 모델(=신경망)이라고 한다.

* 피드 포워드 신경망 : 신호가 되돌아가는 경로가 없는 신경망

입력층을 제외하고, 2층으로 이루어진 2층의 피드 포워드 신경망.(전체는 3층)

2차원의 입력을 3개의 뉴런으로 출력하기 때문에 2차원에서 주어진 수치를 3개의 카테고리로 분류할 수 있다.
__각각의 출력 뉴런의 출력 값이 각각의 카테고리에 속하는 확률을 나타내도록 학습시킨다.__

<br>
i번째 입력부터 j번째 뉴런에 대한 가중치를 w_ji로 쓰고, j번째 뉴런의 입력 총합을 b_j로 한다.

@@@@@@@ 필기

가중치 w_ji의 인덱스가 2개 있다.
입력 총합의 수식에서 동일한 인덱스가 나란히 줄 서는 법칙이 생기고, 행렬 표기에 대응되는 이점이 있다.

@@@@@@@@@@@ 필기

중간층의 출력으로 출력층의 뉴런의 활동이 정해진다.
중간층 j번째 뉴런부터 출력층 k번째 뉴런의 가중치를 v_kj로 나타낸다.
출력층 k번째 뉴런의 입력 총합을 a_k로 한다.

@@@@@@@ 필기

출력층의 출력 y_k는 소프트맥스 함수를 사용하여 나타낸다.

@@@@@@@ 필기

소프트맥스 함수를 사용했기 때문에 y_k의 합이 1이 되어, 확률적 해석이 가능해진다.

=> 일반적인 경우
입력 차원 :D, 중간층 뉴런의 개수: M, 출력 차원: K

@@@@@@@@ 필기

※ 입력 총합을 구할 때, 더미 뉴런의 몫까지 포함하여 D+1, M+1회의 덧셈을 해야 한다는 것을 잊지 말자!

<br>

<br>

### 7.2.2 2층 피드 포워드 신경망의 구현

3클래스 데이터를 사용한 2층 피드 포워드 신경망을 만들어보자.

```python
import numpy as np

# 데이터 생성 --------------------
np.random.seed(seed=1)
N = 200   # 데이터의 수
K = 3     # 분포의 수
T = np.zeros((N, 3), dtype=np.uint8)  # 클래스 데이터 Nx3벡터.
X = np.zeros((N, 2))    #입력 데이터 Nx2벡터.
X_range0 = [-3, 3]    #X0의 범위, 표시용
X_range1 = [-3, 3]    #X1의 범위, 표시용
Mu = np.array([[-.5, .5], [.5, 1.0], [1, -.5]])     # 세 개의 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])    # 세 개의 분포의 분산
Pi = np.array([0.4, 0.8, 1])    # 각 분포에 대한 비율

for n in range(K):
    if wk<Pi[k]:
        T[n, k] = 1
        break
for k in range(2):
    X[n, k] = np.random.randn() * Sig[T[n, :] ==1, k]+Mu[T[n, :]==1, k]
```

이 데이터를 훈련 데이터 X_train, T_train과 테스트 데이터 X_test, T_test로 나누어둔다.
나누는 이유는 오버피팅이 일어나고 있지는 않았는지 확인하기 위함이다.

```python
# ------------- 2분류 데이터를 테스트 훈련 데이터로 분할
TestRatio = 0.5
X_n_training = in(N*TestRatio)
X_train = X[:X_n_training, :]
X_test = X[X_n_training:, :]
T_train = T[:X_n_training, :]
T_test = T[X_n_training:, :]


# ------------- 데이터를 'class_data.npz'에 저장
np.savez('class_data.npz', X_train=X_train, T_train=T_train, X_test=X_test, T_test=T_test, X_range0=X_range0, X_range1=X_range1)
```

분할한 데이터를 그림으로 나타내본다.

```python
import matplotlib.pyplot as plt
%matplotlib inline


# 데이터를 그리기 ------------------------------
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none',
                 marker='o', markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)


# 메인 ------------------------------------
plt.figure(1, figsize=(8, 3.7))
plt.subplot(1, 2, 1)
Show_data(X_train, T_train)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Training Data')
plt.subplot(1, 2, 2)
Show_data(X_test, T_test)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
plt.show()
```

@@@@@@@이미지

`Show_data(x, t)`는 분포 표시용의 함수를 만들고, 훈련 데이터도 테스트 데이터에서 공통으로 사용할 수 있도록 하였다.

<br>

2층의 피드 포워드 신경망을 정의하는 함수를 `FNN`으로 한다.

`FNN`는 네트워크의 입력 __x__ 를 받아 __y__ 를 출력한다. 입력 __x__ 는 D차원 벡터, 출력 __y__ 는 K차원 벡터이다.

* 네트워크 함수
  * __x__ 를 데이터 수 NxD차원의 행렬
  * __y__ 를 데이터 수 NxK차원의 행렬로 한다.
    벡터 __y__ 의 요소 __y[n, 0], y[n, 1], y[n, 2]__ 는 __x[n, :]__ 이 클래스 0, 1, 2에 속할 가능성을 나타낸다. 합은 1이 됨에 주의! (<-소프트맥스함수의 결과이므로.)
  * 중간층의 수와 출력의 차원도 M, K로 하여 일반적인 형태의 네트워크 함수를 표현한다.
  * 네트워크의 동작을 결정하는 중요한 매개 변수인 중간층 가중치 __w__ 와 출력층의 가중치 __v__ 도 네트워크에 전달한다.
    * __w__ 는 Mx(D+1)의 행렬(더미 입력 포함)
    * __v__ 는 Kx(M+1)의 행렬(더미 입력 포함)
    * w와 v의 정보는 w와 v를 한 덩어리로 한 벡터, __wv__ 로 전달한다. 길이 M*(D+1)+K*(M+1)인 벡터.
    -> 학습하는 매개 변수를 한 곳에 모아두면 최적화 프로그램을 만들기 쉽다.
  * 출력 : N개의 데이터에 대응한 출력 __y__ (NxK행렬)와 중간층의 출력 __z__, 출력층과 중간층의 입력총합 __a__, __b__ 도 출력한다.

```python
#시그모이드 함수 -------------
def Sigmoid(x):
    y = 1/(1+exp(-x))
    return y


# 네트워크 -------------------
def FNN(wv, M, K, x):
    N, D = x.shape       #입력 차원 ex) 16*2라면 N에 16, D에 2 저장.
    w = wv[ :M*(D+1)]    #중간층 뉴런의 가중치
    v = wv[M*(D+1): ]    #출력층 뉴런의 가중치
    w.reshape(M, (D+1))
    v.reshape(K, (M+1))
    b = np.zeros((N, M+1))    # 중간층 뉴런의 입력총합
    z = np.zeros((N, M+1))    # 중간층 뉴런의 출력
    a = np.zeros((N, K))      # 출력층 뉴런의 입력총합
    y = np.zeros((N, K))      # 출력층 뉴런의 출력

    for n in range(N):
        #중간층 계산
        for m in range(M):
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1] )  # (A)
                      #항상 1이 되는 더미 입력을 x의 3번째 요소로 덧붙임.
                      # np.r_[A, B]는 행렬을 옆으로 연결시키는 명령어.
            z[n, m] = Sigmoid(b[n, m])
        #출력층 계산
        z[n, M] = 1   #더미 뉴런
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])     # u를 구한다.
        for k in range(K):
            y[n, k] = np.exp(a[n, k])/wkz   # y에 소프트맥스 함수 적용.
    return y, a, z, b


# test -------------
WV = np.ones(15)
M = 2
K = 3
FNN(WV, M, K, X_train[:2, :])
====
(array([[0.33333333, 0.33333333, 0.33333333],
        [0.33333333, 0.33333333, 0.33333333]]),
 array([[2.6971835 , 2.6971835 , 2.6971835 ],
        [1.49172649, 1.49172649, 1.49172649]]),
 array([[0.84859175, 0.84859175, 1.        ],
        [0.24586324, 0.24586324, 1.        ]]),
 array([[ 1.72359839,  1.72359839,  0.        ],
        [-1.12079826, -1.12079826,  0.        ]]))
```

M=2, K=3으로 하여 WV는 길이가 15인 가중치 벡터.
WV의 요소를 모두 1로 하여 입력으로 X_train의 두 데이터만 입력했을 때의 출력은 위와 같다.
위에서부터 각 array는 y, a, z, b의 값이 된다.

`np.r_[A, B]`는 행렬을 옆으로 연결시키는 명령어.
\#(A)에서 x[n, :]와 마지막 더미 입력인 1을 연결시켰다.


<br>

### 7.2.3 수치 미분법

2층의 피드 포워드 네트워크에서 3분류 문제 풀기를 생각한다.
분류 문제 -> 오차 함수로 __평균 교차 엔트로피 오차__ 를 사용한다.

@@@@@@@@@@@ 필기

이 평균 교차 엔트로피 오차를 `CE_FNN` 함수로 구현한다.

```python
# 평균 교차 엔트로피 오차 ---------
def CE_FNN(wv, M, K, x, t):
    N, D = x.shape
    y, a, z, b = FNN(wv, M, K, x)
    ce = -np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / N
    return ce


# test ---
WV = np.ones(15)
M = 2
K = 3
CE_FNN(WV, M, K, X_train[:2, :], T_train[:2, :])
====
1.0986122886681098
```

`CE_FNN`은 `FNN`과 마찬가지로 매개변수 w와 v를 붙인 __wv__ 를 입력한다.
네트워크의 크기를 결정하는 M, K, 입력데이터 x와 목표 데이터 t를 입력한다.
내부에서 FNN이 x에 대한 y를 출력하고, y와 t를 비교하여 크로스 엔트로피가 계산된다.

! 경사 하강법을 사용하려면 오차 함수를 매개변수로 편미분한 식이 필요하지만, 간단히 수치적 미분과 마찬가지의 결과를 내는 식을 사용할 수 있다.
미분의 정의를 사용하여,
> { E(w'+e) - E(w'-e) }/ 2e 가 e가 아주 작은 값일 때 E를 w에 대해 미분한 것과 근사한다.

여러 개의 매개변수에 대응하기 위해, 현재의 w0', w1', w2'라는 점에서 E(w0, w1, w2)의 기울기를 알려면
w1', w2'는 그대로 고정하고 w0에 대한 편미분을 근사한다.
w0', w2'는 그대로 고정하고 w1에 대한 편미분을 근사한다.
w0', w1'는 그대로 고정하고 w2에 대한 편미분을 근사한다.

__이 방법의 단점은 한 매개 변수의 미분을 계산하기 위한 매개 변수 하나에 두 번의 E계산이 필요하다는 것이다.__

`CE_FNN`의 수치 미분을 출력하는 함수 `dCE_FNN_num`을 만든다.

```python
# - 수치 미분 ------------------
def dCE_FNN_num(wv, M, K, x, t):
    epsilon = 0.001           #아주 작은 값. 수치 미분에 사용한다.
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv


#--dVW의 표시 (막대 그래프 )------------------
def Show_WV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:],
            align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)


#-test----
M = 2
K = 3
nWV = M * 3 + K * (M + 1)
np.random.seed(1)
WV = np.random.normal(0, 1, nWV)
dWV = dCE_FNN_num(WV, M, K, X_train[:2, :], T_train[:2, :])
print(dWV)
plt.figure(1, figsize=(5, 3))
Show_WV(dWV, M)
plt.show()
```

@@@@@@@@@@@@@@ 이미지

<br>

### 7.2.4 수치 미분법에 의한 경사 하강법

`Fit_FNN_num`함수로 분류 문제를 경사 하강법으로 풀어보자.
__`Fit_FNN_num(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha)`__
* wv_init : 학습시키는 가중치의 초기값
* n : 학습 단계 수
* alpha : 학습 상수
* 출력 : 최적화된 매개 변수인 wvt


학습 프로그램이 잘 작동한다면, __훈련 데이터의 오차가 단조롭게 감소하여 일정한 값으로 수렴__ 하는 것을 확인할 수 있다.
<u>학습에 사용하지 않은 테스트 데이터의 오차도 도중에 오르지 않고 단순하게 떨어지고 있으면 오버 피팅은 일어나지 않았다고 해석할 수 있다.</u>

가중치 그래프가 교차 : 가중치를 갱신하는 방향 변화(=오차 함수의 기울기의 방향 변화)
가중치가 안장점이라는 지점 근처를 통과했기 때문일지도 모른다.

> 안장점 : 어느 방향은 계곡, 다른 방향으로는 산이 되는 지점. 안장점 근처에서는 학습 속도가 느리며, 어느 정도 학습이 진행되면 방향이 변화하고, 갱신이 가속된다. 안장점은 오차 함수가 최소가 되는 지점이 될 수 없다.

<br>

### 7.2.5 오차 역전파법(Backpropagation)

네트워크의 출력에서 발생하는 오차( yn-tn )의 정보를 사용해서 출력층의 가중치 v_kj에서 중간층의 가중치 w_ji로 입력 방향의 반대로 가중치를 갱신해 나가는 방법.

경사 하강법을 피드 포워드 네트워크에 적용하면 오차 역전파법이 자연스럽게 도출된다.

경사 하강법 적용을 위해 오차 함수를 매개변수로 편미분한다.
네트워크가 클래스 분류를 하므로 오차함수는 평균 교차 엔트로피 오차를 사용하자.

@@@@@@@@@@ 필기

평균 상호 엔트로피 오차(__E(w, v)__)는 데이터 각각의 상호 엔트로피 오차(__En(w, v)__)의 평균으로 해석된다.

경사 하강법에서 사용하는 E의 매개변수의 편미분은, 각 데이터 n에 대한 매개변수의 편미분을 구하여 평균을 하면 구할 수 있다.

@@@@@@@@@@ 필기

네트워크 매개변수는 __w__ 뿐만 아니라 __v__ 도 있다. 우선 En을 v로 편미분한 식을 구하여 En을 w로 편미분한 식을 구하는 순서로 유도하자.

### 7.2.6 En을 v로 편미분한 식 구하기

편미분의 연쇄율을 사용하여 두 미분의 곱으로 분해한다.

@@@@@@ 필기

가중치 v_kj는 중간증(1층)의 뉴런 j에서 출력층(2층)의 뉴련 k에 정보를 전달하는 결합의 가중치이다.
이 결합의 변경 크기는 입력크기 z_j와 그 앞에서 생기는 오차의 곱으로 결정된다.
=> 오차가 없으면 결합을 변경할 필요가 없음을 뜻한다.

목표 데이터 t_k가 0인데 출력 y_k가 0보다 클 경우 오차는 y_k-t_k > 0이 된다.
=> v_kj는 감소하는 방향으로 변경된다. (-a*오차*z_j <0)
=> 출력이 너무 커서 오차가 발생했기 때문에 뉴런 z_j의 영향을 줄이는 방향으로 가중치를 변경시킨다.

<br>

### 7.2.7 En을 w로 편미분한 식 구하기

편미분의 연쇄 법칙을 이용해서 분해한다.

@@@@@@@@@@ 필기

오차 역전파법은 네트워크 계층이 더 늘어도 가중치 매개 변수의 학습 법칙을 도출할 수 있다.

1. 네트워크에 __x__ 를 입력하고 출력 __y__ 를 얻습니다. 이 때 중간에 계산된 __b, z, a__ 도 보유해둔다.
2. 출력 __y__ 를 목표 데이터 __t__ 와 비교해 그 차이(오차)를 계산한다.
이 오차는 출력층의 각 뉴런에 할당된다고 생각한다.
3. 출력층의 오차를 사용하여 중간층의 오차를 계산한다.
4. 결합 본래의 신호 강도와 결합처의 오차 정보를 사용하여 가중치 매개 변수를 갱신한다.

이는 데이터 하나에 대한 갱신이므로 N개의 데이터에 대해 1\~4의 절차를 처리해야한다.

@@@@@@@@@@@@ 이미지

<br>

### 7.2.8 오차 역전파법의 구현

∂E/∂w 및 ∂E/∂v를 구하는 함수 `dCE_FNN`를 만든다.
입력은 `CE_FNN`과 동일하며, ∂E/∂w 및 ∂E/∂v는 dw, dv로 하고, 함수의 출력은 이들은 결합한 __dwv__ 로 한다.

```python
# -- 해석적 미분 -----------------------------------
def dCE_FNN(wv, M, K, x, t):
    N, D = x.shape
    # wv을 w와 v로 되돌림
    w = wv[:M * (D + 1)]
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):]
    v = v.reshape((K, M + 1))
    # ① x를 입력하여 y를 얻음
    y, a, z, b = FNN(wv, M, K, x)
    # 출력 변수의 준비
    dwv = np.zeros_like(wv)
    dw = np.zeros((M, D + 1))
    dv = np.zeros((K, M + 1))
    delta1 = np.zeros(M) # 1층 오차
    delta2 = np.zeros(K) # 2층 오차(k = 0 부분은 사용하지 않음)
    for n in range(N): # (A)
        # ② 출력층의 오차를 구하기
        for k in range(K):
            delta2[k] = (y[n, k] - t[n, k])
        # ③ 중간층의 오차를 구하기
        for j in range(M):
            delta1[j] = z[n, j] * (1 - z[n, j]) * np.dot(v[:, j], delta2)
        # ④ v의 기울기 dv를 구하기
        for k in range(K):
            dv[k, :] = dv[k, :] + delta2[k] * z[n, :] / N
        # ④ w의 기울기 dw를 구하기
        for j in range(M):
            dw[j, :] = dw[j, :] + delta1[j] * np.r_[x[n, :], 1] / N
    # dw와 dv를 합체시킨 dwv로 만들기
    dwv = np.c_[dw.reshape((1, M * (D + 1))), \
                dv.reshape((1, K * (M + 1)))]
    dwv = dwv.reshape(-1)
    return dwv


#------Show VW
def Show_dWV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3],
            align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:],
            align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)


#-- 동작 확인
M = 2
K = 3
N = 2
nWV = M * 3 + K * (M + 1)
np.random.seed(1)
WV = np.random.normal(0, 1, nWV)


dWV_ana = dCE_FNN(WV, M, K, X_train[:N, :], T_train[:N, :])
print("analytical dWV")
print(dWV_ana)


dWV_num = dCE_FNN_num(WV, M, K, X_train[:N, :], T_train[:N, :])
print("numerical dWV")
print(dWV_num)


plt.figure(1, figsize=(8, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
Show_dWV(dWV_ana, M)
plt.title('analitical')
plt.subplot(1, 2, 2)
Show_dWV(dWV_num, M)
plt.title('numerical')
plt.show()
```

@@@@@@@@@@@@@ 이미지

동작 확인을 위해 임의로 생성한 가중치 매개 변수 WV에 대한 해석적 미분값 dWV_ana를 출력하고, 이전에 작성한 수치 미분값도 출력해보면 두 값이 거의 일치함을 볼 수 있다.

수치 미분으로 풀었던 분류 문제를 오차 역전파법으로 풀어보자.
함수 __`Fit_FNN`__ 은 수치 미분의 경우 `Fit_FNN_num`과 거의 동일하고, 수치 미분을 사용했던 부분을 방금 작성한 `dCE_FNN`으로 교체했다.

```python
import time


# 해석적 미분을 사용한 구배법 -------
def Fit_FNN(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    wv = wv_init.copy()
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n, len(wv_init)))
    epsilon = 0.001
    for i in range(n):
        wv = wv - alpha * dCE_FNN(wv, M, K, x_train, t_train) # (A)
        err_train[i] = CE_FNN(wv, M, K, x_train, t_train)
        err_test[i] = CE_FNN(wv, M, K, x_test, t_test)
        wv_hist[i, :] = wv
    return wv, wv_hist, err_train, err_test


# 메인 ---------------------------
startTime = time.time()
M = 2
K = 3
np.random.seed(1)
WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
N_step = 1000
alpha = 1
WV, WV_hist, Err_train, Err_test = Fit_FNN(
    WV_init, M, K, X_train, T_train, X_test, T_test, N_step, alpha)
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))
====
Calculation time:17.983 sec
```

수치 미분에 비하면 계산 시간이 훨씬 짧다.

결과를 표시한다.
```python
plt.figure(1, figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)
# 학습 오차의 표시 ---------------------------
plt.subplot(1, 3, 1)
plt.plot(Err_train, 'black', label='training')
plt.plot(Err_test, 'cornflowerblue', label='test')
plt.legend()
# 가중치의 시간 변화 표시 ---------------------------
plt.subplot(1, 3, 2)
plt.plot(WV_hist[:, :M * 3], 'black')
plt.plot(WV_hist[:, M * 3:], 'cornflowerblue')
# 경계선 표시 --------------------------
plt.subplot(1, 3, 3)
Show_data(X_test, T_test)
M = 2
K = 3
show_FNN(WV, M, K)
plt.show()
```

@@@@@@@@@@@ 이미지

수치 미분 때와 거의 동일한 결과를 얻을 수 있다.
네트워크의 규모가 커질수록 미분 계산 속도가 빠른 오차 역전파법을 사용하는 것이 좋다.
하지만, 수치 미분은 도출된 미분 방정식이 맞는지 확인하는 용도로 잘 사용된다. 앞으로 새로운 오차 함수의 미분 방정식을 구할 경우에 먼저 수치 미분으로 올바른 값을 구해두는 것이 좋다.

<br>

### 7.2.9 학습 후 뉴런의 특성

b_j, z_j, a_k, y_k의 특성을 그림으로 나타내서 확인해보자.

```python
from mpl_toolkits.mplot3d import Axes3D


def show_activation3d(ax, v, v_ticks, title_str):
    f = v.copy()
    f = f.reshape(xn, xn)
    f = f.T
    ax.plot_surface(xx0, xx1, f, color='blue', edgecolor='black',
                    rstride=1, cstride=1, alpha=0.5)
    ax.view_init(70, -110)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks(v_ticks)
    ax.set_title(title_str, fontsize=18)


M = 2
K = 3
xn = 15 # 등고선 표시 해상도
x0 = np.linspace(X_range0[0], X_range0[1], xn)
x1 = np.linspace(X_range1[0], X_range1[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
x = np.c_[np.reshape(xx0, xn * xn, 1), np.reshape(xx1, xn * xn, 1)]
y, a, z, b = FNN(WV, M, K, x)


fig = plt.figure(1, figsize=(12, 9))
plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95,
                    top=0.95, wspace=0.4, hspace=0.4)

for m in range(M):
    ax = fig.add_subplot(3, 4, 1 + m * 4, projection='3d')
    show_activation3d(ax, b[:, m], [-10, 10], '$b_{0:d}$'.format(m))
    ax = fig.add_subplot(3, 4, 2 + m * 4, projection='3d')
    show_activation3d(ax, z[:, m], [0, 1], '$z_{0:d}$'.format(m))


for k in range(K):
    ax = fig.add_subplot(3, 4, 3 + k * 4, projection='3d')
    show_activation3d(ax, a[:, k], [-5, 5], '$a_{0:d}$'.format(k))
    ax = fig.add_subplot(3, 4, 4 + k * 4, projection='3d')
    show_activation3d(ax, y[:, k], [0, 1], '$y_{0:d}$'.format(k))


plt.show()
```

@@@@@@@@@@@@@@@ 이미지

입력 총합 b_j는 입력 x_i의 선형 합이므로 입출력 맵이 평면이 된다.
입력 총합 b_j가 시그모이드 함수를 빠져나가면 출력 z_j가 된다.

출력층의 입력 총합 a_k의 입출력 맵은 z_0, z_1의 두 입출력 맵의 선형 합으로 이루어진다.
-  ex) a_1의 맵은 1.2*z_0 + 5.5*z_1 = 3.2가 만들어짐.

a_k는 소프트맥스 함수를 지나 0\~1의 범위로 뭉개져서 y_k가 만들어진다.
y_0, y_1, y_2가 솟아 오르는 부분은 각 클래스 0, 1, 2로 분류되는 범위에 대응된다.
0, 1, 2의 면을 모두 더하면 높이가 1인 평면이 된다.

출력층은 중간층 뉴런을 기저 함수로 선형 소프트맥스 모델로 간주될 수 있다. 하지만, '신경망 모델의 경우는 기저 함수의 특성도 학습에 의해 자동으로 최적화되는 특별한 선형 소프트맥스 모델'이다.

<br>

<br>

## 7.3 케라스로 신경망 모델 구현

신경망의 다양한 라이브러리가 구현되어있으므로 그걸 사용해보자.
2015년에 출시된 케라스 라이브러리.
케라스 라이브러리를 사용하면 텐서플로를 쉽게 동작시킬 수 있다.

### 7.3.1 2층 피드 포워드 신경망

필요한 라이브러리를 import하고 저장된 데이터를 load한다.

```python
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1) # (A) 케라스 내부에서 사용되는 난수 초기화하는 코드
import keras.optimizers # (B) 케라스 관계의 라이브러리 import
from keras.models import Sequential # (C) 케라스 관계의 라이브러리 import
from keras.layers.core import Dense, Activation #(D) 케라스 관계의 라이브러리 import


# 데이터 로드 ---------------------------
outfile = np.load('class_data.npz')
X_train = outfile['X_train']
T_train = outfile['T_train']
X_test = outfile['X_test']
T_test = outfile['T_test']
X_range0 = outfile['X_range0']
X_range1 = outfile['X_range1']
```

이전에 정의했던 데이터를 그림으로 그리는 함수 재정의.

```python
# 데이터를 그리기 ------------------------------
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none', marker='o',
                 markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)
```

케라스를 사용해서 2층 피드백 신경망 모델을 만들고 학습시킨다.

```python
# 난수 초기화
np.random.seed(1)


# --- Sequential 모델 작성
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid',
                kernel_initializer='uniform')) # (A)
model.add(Dense(3,activation='softmax',
                kernel_initializer='uniform')) # (B)
sgd = keras.optimizers.SGD(lr=1, momentum=0.0,
                           decay=0.0, nesterov=False) # (C)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy']) # (D)


# ---------- 학습
startTime = time.time()
history = model.fit(X_train, T_train, epochs=1000, batch_size=100,
                    verbose=0, validation_data=(X_test, T_test)) # (E)


# ---------- 모델 평가
score = model.evaluate(X_test, T_test, verbose=0) # (F)
print('cross entropy {0:3.2f}, accuracy {1:3.2f}'\
      .format(score[0], score[1]))
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))
====
cross entropy 0.26, accuracy 0.90
Calculation time:2.396 sec
```

케라스 라이브러리를 이용해서 이전 계산보다 더 빠르게 답을 얻을 수 있다.

<br>

### 7.3.2 케라스 사용의 흐름

케라스에서 필요한 라이브러리를 import 한다.

```python
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
```


__Sequential__ 이라는 유형의 네트워크 모델로 __model__ 을 만든다.

```python
model = Sequential()
```

이 모델은 변수가 아니고, 'Sequential 클래스에서 생성된 객체'이다.
케라스는 이 model에 층을 추가해 네트워크의 구조를 정의한다.


이 model에 중간층으로 __Dense__ 라는 전결합형의 층을 추가한다.

```python
model.add(Dense(2, input_dim=2, activation='sigmoid', kernel_initializer='uniform'))  # (A)
```
`Dense()`
첫 번째 인수 2 : 뉴런의 수
두 번째 인수 input_dim : 입력의 차원이 2임
세 번째 인수 activation : 활성화 함수로 시그모이드 함수를 사용함
네 번째 인수 kernel_initializer : 가중치 매개 변수의 초기값을 균일 난수에서 결정함.
더미 입력은 기본적으로 설정되어 있다.

출력층도 `Dense()`로 마찬가지로 정의한다.

```python
model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))   # (B)
```
`Dense()`
첫 번째 인수 3 : 뉴런의 수
두 번째 인수 activation : 활성화함수로 소프트맥스 함수 사용함
세 번째 인수 kernel_initializer : 가중치 매개변수를 균일 난수로 한다.


학습 방법의 설정을 `keras.optimizers.SGD()`에서 실시해서 그 내용을 __sgd__ 에 넣는다.

```python
sgd = keras.optimizers.SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)   # (C)
```

lr : 학습 속도
이 sgd를 model.compile()에 전달해서 학습 방법의 설정이 이루어진다.

```python
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])   # (D)
```

loss='categorical_crossentropy' : 목적 함수를 교차 엔트로피 오차로 지정한다
metrics=['accuray'] : 학습의 평가로 정답률도 계산하도록 지시한다.
  정답률: 예측의 확률이 가장 높은 클래스를 예측할 때 모든 데이터에 대해 몇 %가 정답인지의 비율

```python
model.fit(X_train, T_train, batch_size=100, epochs=1000, verbose=0, validation_data=(X_test, T_test))   # (E)
```

`model.fit`의 인수
X_train, T_train : 훈련 데이터 지정
batch_size=100 : 1단계 분의 기울기를 계산하는데 사용하는 학습 데이터의 수
epochs=1000 : 전체 데이터를 학습에 사용한 횟수
verbose=0 : 학습 진행 상황을 표시하지 않은
validation_data=(X_test, T_test) : 평가용 데이터 지정
출력 : history에 학습 과정의 정보가 담겨있다.

```python
# ----------- 모델 평가
score = model.evaluate(X_test, T_test, verbose=0)   # (F)
print('loss {0:f}, acc {1:f}'.format(score[0], score[1]))
```

`model.evaluate()`로 최종 학습의 평가 값을 출력.
score[0] : 테스트 데이터의 상호 엔트로피 오차
score[1] : 테스트 데이터의 정답률

<br>

학습 과정과 그 결과를 그래프로 표시한다.

```python
plt.figure(1, figsize = (12, 3))
plt.subplots_adjust(wspace=0.5)


# 학습 곡선 표시 --------------------------
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], 'black', label='training') # (A) 훈련 데이터의 교차 엔트로피 오차의 시계열 정보
plt.plot(history.history['val_loss'], 'cornflowerblue', label='test') # (B) 훈련 데이터의 교차 엔트로피 오차
plt.legend()


# 정확도 표시 --------------------------
plt.subplot(1, 3, 2)
plt.plot(history.history['acc'], 'black', label='training') # (C) 훈련 데이터의 정답률
plt.plot(history.history['val_acc'], 'cornflowerblue', label='test') # (D) 테스트 데이터의 정답률
plt.legend()


# 경계선 표시 --------------------------
plt.subplot(1, 3, 3)
Show_data(X_test, T_test)
xn = 60 # 등고선 표시 해상도
x0 = np.linspace(X_range0[0], X_range0[1], xn)
x1 = np.linspace(X_range1[0], X_range1[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
x = np.c_[np.reshape(xx0, xn * xn, 1), np.reshape(xx1, xn * xn, 1)]
y = model.predict(x) # (E) 학습이 완료된 모델에 의한 임의의 입력 x에 대한 예측
K = 3
for ic in range(K):
    f = y[:, ic]
    f = f.reshape(xn, xn)
    f = f.T
    cont = plt.contour(xx0, xx1, f, levels=[0.5, 0.9], colors=[
        'cornflowerblue', 'black'])
    cont.clabel(fmt='%1.1f', fontsize=9)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
plt.show()
```

@@@@@@@@@@@@@@@@@@ 이미지

(A)에서 훈련 데이터의 오차가 빠르게 감소함을 볼 수 있다. 테스트 데이터의 오차도 증가하고 있지 않기 때문에 오버피팅이 일어나지 않았다고 볼 수 있다.

정답률은 학습이 잘 이루어지면 1로 다가가는데, 때로는 감소가 일어날 수 있다.
