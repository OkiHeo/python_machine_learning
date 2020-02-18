* 비지도 학습 : 입력 정보만 사용하는 학습 방법.
  * 클러스터링 (Clustering)
  * 차원 압축 (Dimensionality reduction)
  * 이상 감지 (Anomaly Detction)

<br>

## 9.1 2차원 입력 데이터

2차원 입력 데이터 __X__ 를 사용하지만, 비지도 학습 문제에서는 세트로 되어 있는 클래스 데이터 T는 사용하지 않는다.
_클래스 정보 없이, 입력 데이터가 비슷한 것끼리 클래스로 나눈 것이 '클러스터링'이다._

데이터 분포의 모양을 클러스터라고 하고, 데이터 분포에서 클러스터를 찾아, 동일한 클러스터에 속하는 데이터 점에는 같은 클래스(라벨)을 붙이고, 다른 클러스터에 속하는 데이터 점에는 다른 클래스를 할당하는 것이 클러스터링이다.

=> 클러스터는 분포의 특징을 나타낸다.
동일한 클러스터에 속하는 데이터 점은 '닮았다'고 간주되며, 다른 클러스터에 속하는 점은 '닮지 않은'것으로 볼 수 있다.

* 클러스터링 알고리즘
  1. k-means 기법
  2. 가우시안 혼합 모델을 사용한 클러스터링


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline



#----------- 데이터 생성
np.random.seed(1)
N = 100         # 생성할 데이터의 개수
K = 3           # 클래스의 개수

T3 = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
X_col = ['cornflowerblue', 'black', 'white']            # 각 클래스를 나타낼 색을 지정

Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])        # 각 클래스 분포의 중심을 나타냄.
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])          # 각 클래스 분포의 분산을 나타냄.
Pi = np.array([0,4, 0.8, 1])                            # 각 클래스에 속할 누적 확률을 나타냄.

for n in range(N):
    wk = np.random.rand()
    # n개의 생성된 랜덤 데이터에 대해 1-of-K 부호화하여 T3에 저장
    for k in range(K):
        if wk<Pi[k]:
            T3[n, k] = 1
            break

    # 2차원 데이터 X의 [n, 0]: x좌표, [n, 1]: y좌표에 분포의 중심, 분산을 이용해서 위치 할당.
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[ T3[n, :]==1, k ] + Mu[ T3[n, :]==1, k ])




# ----------------- 데이터 그림으로 나타내기
def show_data(x):
    plt.plot( x[:, 0], x[:, 1], linestyle='none', marker='o', markersize=6, markeredgecolor='black', color='gray', alpha=0.8)
    plt.title('Generated Data')
    plt.grid(True)



# -------------------- 메인
plt.figure(1, figsize=(4, 4))
show_data(X)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()

np.savez('data_ch9.npz', X=X, X_range0=X_range0, X_range1=X_range1)     # 생성된 데이터를 파일에 저장. 나중에 다시 사용하기 위해서!
```

@@@@@@@@@@@이미지

<br>

<br>

## 9.2 K-means 기법

## 9.2.1 K-means 기법의 개요

1. step 0 : 데이터 분포 중심값(μ)에 초기값을 부여한다.
2. step 1 : 분포의 중심값(μ)으로 분산값(R)을 갱신한다.
  각 데이터 점을 가장 중심이 가까운 클러스터에 소속시킨다.
3. step 2 : 분산값(R)으로 분포의 중심값(μ)을 갱신한다.
  각 클러스터에 속하는 데이터 점의 중심을 새로운 분포의 중심값으로 한다.

데이터 분포의 중심값(μ)이 더이상 변하지 않을 때까지 step 1, step 2를 반복한다.


* __μ__ : 클러스터의 중심 벡터.
  클러스터의 중심을 나타낸다.
* __R__ : 클래스 지시 변수.
  각 데이터 점이 어떤 클러스테 속하는지를 나타낸다.

<br>

### 9.2.2 Step 0: 변수의 초기화

__μ__\_k = [μ_k0, μ_k1] (k=0, 1, 2)

클래스 지시 변수 __R__ 은 각 데이터가 어느 클래스에 속해 있는지를 1-of-K부호화법으로 나타낸 행렬이다.
r\_nk = 1 (데이터 n이 k에 속하는 경우)
r\_nl = 0 (데이터 n이 k에 속하지 않는 경우)

```python
Mu = np.array([[-2, 1], [-2, 0], [-2 ,-1]])
R = np.c_[np.ones((N, 1)), np.zeros((N, 2), dtype=int)]     # 모든 데이터가 클래스 0에 속하도록 초기화한다.  
```

```python
# ------------- 데이터를 그리는 함수
def show_prm(x, r, mu, col):
    for k in range(K):
        # 데이터 분포의 표시
        plt.plot(x[r[:, k]==1, 0], x[r[:, k]==1, 1], marker='o', markerfacecolor=X_col[k], markeredgecolor='k', markersize=6, alpha=0.5, linestyle='none')
        # 데이터 평균을 '★'로 표시
        plt.plot(mu[k, 0], mu[k, 1], marker='*', markerfacecolor=X_col[k], markersize=15, markeredgecolor='k', markeredgewidth=1)

    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.grid(True)


# -----------------------------------
plt.figure(figsize=(4, 4))
R = np.c_[np.ones((N, 1)), np.zeros((N, 2))]        # 초기값, 전부 0클래스에 속함.
show_prm(X, R, Mu, X_col)
plt.title('initial Mu and R')
plt.show()
```

@@@@@@@@@이미지

### 9.2.3 Step 1: R의 갱신

갱신 방법 : "각 데이터 점을 가장 중심이 가까운 클러스터에 넣는다."

```python
# r을 정한다. (step1 )----------------
def step1_kmeans(x0, x1, mu):
    N = len(x0)
    r = np.zeros((N, K))
    for n in range(N):      # N개의 데이터에 대해 반복.
        wk = np.zeros(K)    
        for k in range(K):  # K개의 클래스에 대해 반복. 어느 클래스와의 중심거리가 가장 짧은가?
            wk[k] = (x0[n]-mu[k, 0])**2 + (x1[n]-mu[k, 1])**2
        r[n, np.argmin(wk)]=1       # n, wk 중 가장 작은 값을 가지는 열에 1을 저장.
    return r



# ---------------------------------------
plt.figure(figsize=(4, 4))
R = step1_kmeans(X[:, 0], X[:, 1], Mu)
show_prm(X, R, Mu, X_col)
plt.title('Step 1')
plt.show()
```

@@@@@@@@@@@@@ 이미지

<br>

### 9.2.4 Step2: μ의 갱신

갱신 방법 : "각 클러스터에 속하는 데이터 점의 중심을 새로운 μ로 한다."

```python
# Mu 결정 (Step 2) ---------------
def step2_kmeans(x0, x1, r):
    mu = np.zeros((K, 2))
    for k in range(K):      # K개의 클래스에 대해
        mu[k, 0] = np.sum(r[:, k]*x0) / np.sum(r[:, k])         # 클래스 k에 속하는 x좌표합 / 클래스 k에 속하는 x좌표개수
        mu[k, 1] = np.sum(r[:, k]*x1) / np.sum(r[:, k])         # 클래스 k에 속하는 y좌표합 / 클래스 k에 속하는 y좌표개수
    return mu


# --------------------------------수
plt.figure(figsize=(4, 4))
Mu = step2_kmeans(X[:, 0], X[:, 1], R)
show_prm(X, R, Mu, X_col)
plt.title('Step 2')
plt.show()
```

@@@@@@@@@@@@@@@이미지

이제 step1과 step2를 반복하며 변수의 값이 변화하지 않으면 프로그램을 종료한다.

<br>

<br>

### 9.2.5 왜곡 척도

K-means 기법의 경우, 데이터 점이 속한 클러스터의 중심까지의 제곱 거리를 전체 데이터로 합한 것이 목적 함수에 대응한다.
이를 왜곡척도라고 한다.

@@@@@@@@@@@@@@ 필기

왜곡 척도를 계산하는 함수 `distortion_measure()`를 정의한다.
R과 Mu를 초기값으로 되돌린다.

```python
# 목적 함수 -------------------
def distortion_measure(x0, x1, r, mu):
    # 입력은 2차원으로 제한하고 있다.
    N = len(x0)
    J = 0

    for n in range(N):
        for k in range(K):
            J = J + r[n, k]*((x0[n]-mu[k, 0])**2 + (x1[n]-mu[k, 1])**2)
    return J


# test ----------------------------
# Mu 와 R을 초기화
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]
distortion_measure(X[:, 0], X[:, 1], R, Mu)
====
771.7091170334878
```

이 함수를 이용해서 K-means 기법의 반복에 의한 왜곡 척도를 계산한다.

```python
# Mu와 R의 초기화
N = X.shape[0]
K = 3

Mu = np.array([[-2, 1], [-2 ,0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]

max_it = 10
it = 0

DM = np.zeros(max_it)       # 왜곡 척도의 계산 결과 저행

# K-means 기법 10번 반복 수행
for it in range(0, max_it):
    R = step1_kmeans(X[:, 0], X[:, 1], Mu)  # R의 갱신. 가장 중심이 가까운 클러스터로..
    DM[it] = distortion_measure(X[:, 0], X[:, 1], R, Mu)        # 왜곡 척도 계산.
    Mu = step2_kmeans(X[:, 0], X[:, 1], R)  # Mu의 갱신. 각 클러스터에 속하는 중심값 계산

print(np.round(DM, 2))
plt.figure(2, figsize=(4, 4))
plt.plot(DM, color='black', linestyle='-', marker='o')
plt.ylim(40, 80)
plt.grid(True)
plt.show()
```

@@@@@@@@@@@@@@ 이미지

Step을 진행할수록 왜곡척도가 점차 감소하고 있다. 곧, Mu와 R의 값이 변하지 않음을 뜻한다.

K-means 기법으로 얻을 수 있는 해는 초기값 의존성이 있다. 처음 Mu에 무엇을 할당하는지에 따라 결과가 달라질 수 있다. 실제로는 다양한 Mu에서 시작하여 얻은 결과 중에 가장 왜곡 척도가 작은 결과를 사용하는 방법을 사용한다.

<br>

<br>

## 9.3 가우시안 혼합 모델

### 9.3.1 확률적 클러스터링

K-means 기법은 데이터 점을 반드시 클러스터에 할당한다.
하지만, '데이터 점 A는 확실히 클러스터 0에 속하지만, 데이터 점 B는 클러스터 0과 1에 모두 속해 있다'고 모호성을 포함해 수히화할 경우는????
=> 확률의 개념을 도입한다.

@@@@@@@ 필기


n번째 데이터가 k에 속할 확률은 아래와 같이 나타낸다.

> γ\_nk = [γ\_n0, γ\_n1, γ\_n2]

관찰은 못했지만 데이터에 영향을 준 변수를 '잠재 변수' 또는 '숨은 변수'라고 한다.

이 잠재 변수를 3차원 벡터를 사용해서 1-of-k부호화법으로 표현할 수 있다.

> z\_n = [z\_n0, z\_n1, z\_n2]

데이터 n이 클래스 k에 속한다면 z\_nk만 1을 취하고 다른 요소는 0으로 한다.

이 관점에서 데이터 n이 '클러스터 k에 속할 확률 γnk'란 데이터 xn인 곤충이 '클래스 k의 변종일 확률'을 의미한다.

> γ\_nk = P(z\_nk=1 | x\_n)

단적으로 말하면, "관찰할 수 없는 Z의 추정치가 γ이다."라고 말할 수 있다.
γ는 "어떤 클러스터에 얼마나 기여하고 있는가"라는 의미에서 부담률(responsibility)라고 한다.

=> __확률적 클러스터링은 데이터의 배후에 숨어 있는 잠재 변수 Z를 확률적으로 γ으로 추정하는 것__ 이다.

<br>

### 9.3.2 가우시안 혼합 모델

부담률 γ를 구하기 위해 가우시안 혼합 모델이라는 확률 모델을 살펴보자.

@@@@@@@@@@@@@@ 필기

가우시안 혼합 모델을 나타내는 함수를 만들어보자.

```python
### 앞에서 사용했던 X, X_range0, X_range1은 그대로 사용한다. load하자.
import numpy as np
wk = np.load('data_ch9.npz')

X = wk['X']
X_range0 = wk['X_range0']
X_range1 = wk['X_range1']
```

가우스 함수 `gauss(x, mu, sigma)`를 정의한다.

```python
# 가우스 함수 ------------------------
# x : NxD 행렬
# mu : 길이 D인 중심 벡터
# sigma : DxD의 공분산 행렬

def gauss(x, mu, sigma):
    N, D = x.shape          # x는 NxD의 행렬.
    c1 = 1/(2*np.pi)**(D/2)
    c2 = 1/ (np.linalg.det(sigma)**(1/2))
    inv_sigma = np.linalg.inv(sigma)
    c3 = x-mu
    c4 = np.dot(c3, inv_sigma)
    c5 = np.zeros(N)

    for d in range(D):
        c5 = c5 + c4[:, d] * c3[:, d]

    p = c1 * c2 * np.exp(-c5/2)
    return p
```

<br>

가우스 함수를 여러 번 더해서 가우시안 혼합 모델 `mixgauss(x, pi, mu, sigma)`를 정의한다.

```python
# 가우시안 혼합 모델 --------------------------------
# x : NxD 행렬
# pi : 혼합 계수. 길이 K의 벡터
# mu : KxD의 행렬. K개의 가우스 함수의 중심을 한 번에 지정함.
# sigma : 공분산 행렬. KxDxD의 3차원 배열 변수로 K개의 가우스 함수의 공분산 행렬을 한꺼번에 지정.
def mixgauss(x, pi, mu, sigma):
    N, D = x.shape
    K = len(pi)
    p = np.zeros(N)

    for k in range(K):
        p = p + pi[k]*gauss(x, mu[k, :], sigma[k, :, :])
    return p
```

<br>

가우시안 혼합 모델을 그래픽으로 표시하는 함수를 만들자.
1. 등고선 표시 함수 `show_contour_mixgauss()`
2. 3D로 나타내는 함수 `show3d_mixgauss()`

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
%matplotlib inline

# 혼합 가우스 등고선 표시 ----------------------------
def show_contour_mixgauss(pi, mu, sigma):
    xn = 40         # 등고선 표시 해상도
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)

    xx0, xx1 = np.meshgrid(x0, x1)

    x = np.c_[np.reshape(xx0, xn*xn, 1), np.reshape(xx1, xn*xn, 1)]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    f = f.T         # transpose
    plt.contour(x0, x1, f, 10, colors='gray')



# 혼합 가우스 3D 표시 ----------------------------------
def show3d_mixgauss(ax, pi, mu, sigma):
    xn = 40         # 해상도
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)

    xx0, xx1 = np.meshgrid(x0, x1)

    x = np.c_[np.reshape(xx0, xn*xn, 1), np.reshape(xx1, xn*xn, 1)]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    f = f.T         # transpose
    ax.plot_surface(xx0, xx1, f, rstride=2, cstride=2, alpha=0.3, color='blue', edgecolor='black')
```

예시를 확인해보자.

```python
# test ---------------------------------------
pi = np.array([0.2, 0.4, 0.4])
mu = np.array([[-2, -2], [-1, -1], [1.5, 1]])
sigma = np.array([[[.5, 0], [0, .5]],
                  [[1, 0.25], [0.25, .5]],
                  [[.5, 0], [0, .5]]])

Fig = plt.figure(1, figsize=(8, 3.5))
Fig.add_subplot(1, 2, 1)
show_contour_mixgauss(pi, mu, sigma)
plt.grid(True)


Ax = Fig.add_subplot(1, 2, 2, projection='3d')
show3d_mixgauss(Ax, pi, mu, sigma)
Ax.set_zticks([0.05, 0.10])
Ax.set_xlabel('$x_0$', fontsize=14)
Ax.set_ylabel('$x_1$', fontsize=14)
Ax.view_init(40, -100)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()
```

@@@@@@@@@@@@@@@@@@@@@이미지

<br>

### 9.3.3 EM 알고리즘의 개요

> EM 알고리즘(expectation-maximization algorithm)
알고리즘 기댓값 최대화 알고리즘이다.
관측되지 않는 잠재변수에 의존하는 확률 모델에서 최대 가능도나 최대 사후확률을 찾는 모수의 추정값을 찾는 반복적인 알고리즘.

가우시안 혼합 모델을 사용하여 데이터의 클러스터링을 수행한다.
EM 알고리즘을 사용하여 가우시안 혼합 모델을 데이터에 피팅해보고, 부담률 γ를 구하는 방법을 알아보자.

1. STEP 1:
  π, μ, Σ에 초기값 부여
2. STEP 2(E STEP):
  π, μ, Σ로 γ를 갱신
  각 데이터 점은 각 클러스터에서 생성된 사후확률로 γ를 계산.
3. STEP 3(M STEP):
  γ로 π, μ, Σ를 갱신
  각 클러스터에 대한 부담률로 각 클러스터의 매개변수를 계산.


K-means 기법에서는 각 클러스터를 중심 벡터 μ로 특정했지만, 가우시안 혼합 보델은 중심벡터뿐만 아니라 공분산 행렬 Σ에 의해 각 클러스터의 확산 정도를 기술한다. 또한 혼합 계수 π에 의해 각 클러스터의 크기 차이를 설명한다.

가우시안 혼합 모델의 출력은 부담률 γ이다.

### 9.3.4 STEP 0: 변수의 준비 및 초기화

```python
# 초기 설정 ---------------------------------
N = X.shape[0]
K = 3

Pi = np.array([0.33, 0.33, 0.34])               #클러스터 혼합 계수
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])     #클러스터 중심 벡터
Sigma = np.array([[[1, 0], [0, 1]],             #클러스터 공분산 행렬
                   [[1, 0], [0, 1]],
                   [[1, 0], [0, 1]]])
Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]


X_col = np.array([[0.4, 0.6, 0.95], [1, 1, 1], [0, 0, 0]])


# 데이터를 그림으로 나타내자 -------------------
def show_mixgauss_prm(x, gamma, pi, mu, sigma):
    N, D = x.shape
    show_contour_mixgauss(pi, mu, sigma)

    for n in range(N):
        col = gamma[n, 0]*X_col[0] + gamma[n, 1]*X_col[1] + gamma[n, 2]*X_col[2]
        plt.plot(x[n, 0], x[n, 1], 'o', color=tuple(col), markeredgecolor='black',
                 markersize=6, alpha=0.5)

    for k in range(K):
        plt.plot(mu[k, 0], mu[k, 1], marker='*',
                 markerfacecolor=tuple(X_col[k]), markersize=15,
                 markeredgecolor='k', markeredgewidth=1)

    plt.grid(True)


plt.figure(1, figsize=(4, 4))
show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)
plt.show()
```

@@@@@@@@@@@@@ 이미지

초기값으로 할당한 중심 벡터가 인접해있기 때문에 3개의 가우스 함수가 겹쳐서 세로로 긴 산과 같은 분포가 형성되어 있다.

<br>

### 9.3.5 STEP 1(E STEP): γ 갱신
