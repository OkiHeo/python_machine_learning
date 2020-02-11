## 4.1 벡터
* 세로 벡터
* 가로 벡터
벡터를 구성하는 숫자 하나하나를 __요소__ 라고 한다.
벡터가 가지는 요소의 수를 벡터의 __차원__ 이라고 한다.
<br>
벡터의 전치 : __`T`__ 로 표현한다.
세로 벡터 <-> 가로 벡터 변환.
<br>
### 4.1.2 파이썬으로 벡터를 정의하기
벡터를 사용하려면 `import numpy`로 numpy라이브러리를 import한다.

```python
#리스트 4-1-(1)
import numpy as np
```

```python
#리스트 4-1-(2)
a = np.array([2,1])
print(a)
print(type(a))
====
[2 1]
<class 'numpy.ndarray'>
```

<br>

### 4.1.3 세로 벡터를 나타내기
1차원 ndarray형은 가로, 세로를 구분하지 않고 __항상 가로 벡터로 표시__ 된다.

특별한 형태의 2차원 ndarray로 세로 벡터를 나타낼 수 있다.

```python
#리스트 4-1-(4)
b = np.array([[1,2], [3,4]])
print(b)
print(type(b))
====
[[1 2]
 [3 4]]
<class 'numpy.ndarray'>
```

<br>

### 4.1.4 전치를 나타내기
전치는 __`변수명.T`__ 로 나타낸다.

```python
#리스트 4-1-(6)
b = np.array([[2], [1]])
print(b)
print(b.T)
====
[[2]
 [1]]
[[2 1]]
```

<br>

### 4.1.5 덧셈과 뺄셈
벡터의 덧셈 a+b는 각 요소를 더하면 된다.
== 두 벡터를 변으로 하는 평행사변형의 대각선을 구하는 연산

```python
#리스트 4-1-(7)
a = np.array([2, 1])
b = np.array([1, 3])
print(a+b)
====
[3 4]
```

벡터의 뺄셈도 마찬가지로 각 요소를 뺀다.
== a와 -b를 변으로 하는 평행사변형의 대각선을 구하는 연산

```python
#리스트 4-1-(8)
a = np.array([2, 1])
b = np.array([1, 3])
print(a-b)
====
[ 1 -2]
```

<br>

### 4.1.6 스칼라의 곱셈
스칼라에 벡터를 곱하면 스칼라 값을 각 요소에 곱하게 된다.

```python
# 리스트 4-1-(9)
print(2*a)
====
[4 2]
```

<br>

### 4.1.7 내적
벡터에는 __내적__ 이라는 곱셈 연산이 존재한다.
_같은 차원_ 을 가진 두 벡터 간의 연산에서 __`·`__ 으로 나타낸다.
__대응하는 요소들을 곱한 뒤 더한 값__ 이 결과값이다.
파이썬에서는 __`변수명1.dot(변수명2)`__ 로 내적을 계산할 수 있다.

```python
# 리스트 4-1-(10)
b = np.array([1, 2])
c = np.array([5, 6])
print(b.dot(c))
====
17        # 1*5 + 2*6 = 17
```

벡터 b를 c로 투영한 벡터를 b\`라고 한다면, b\`와 c의 길이를 곱한 값이 내적이다.
내적은 "두 벡터가 비슷한 방향을 향할 때" 큰 값을 가진다.
반대로, "두 벡터가 수직에 가까울수록" 작은 값을 가진다. 수직일 때의 내적값은 0이다.
=> __내적은 두 벡터의 유사성과 관련있다.__

<br>

### 4.1.8 벡터의 크기
수식으로 벡터 a의 크기는 |a|로 나타낸다.
파이썬에서는 __`np.linalg.norm()`__ 으로 벡터의 크기를 구할 수 있다.

vector [a, b]의 크기 = $\sqrt(a^2+b^2)$

```python
# 리스트 4-1-(11)
a = np.array([3, 4])
print(np.linalg.norm(a))
====
5.0
```

<br>

<br>

## 4.2 합의 기호

for문을 이용해서 나타낼 수 있다.

### 4.2.1 합의 기호가 들어간 수식을 변형시키기

식에 변수가 존재하지 않는 경우에 아래와 같이 쉽게 변형할 수 있다.

$\sum_{i=1}^5 3$ = 3 $\times$5 = 15

f(n) = 스칼라$\times$n의 함수인 경우에는 스칼라를 $\Sigma$ 기호 밖으로 낼 수 있다.

$\sum_{n=1}^3 2n^2$ = 2$\sum_{n=1}^3 n^2$

<br>

벡터의 내적도 합의 기호를 사용해서 작성할 수 있다.

w = [$w_0$ $w_1$ ... $w_{D-1}$]$^T$,

x = [$x_0$ $x_1$ ... $x_{D-1}$]$^T$

w와 x의 내적은 아래와 같다.

w$\cdot$x = $w_0x_0$ + $w_1x_1$ + ... + $w_{D-1}x_{D-1}$ = $\sum_{i=0}^{D-1} w_ix_i$

좌측은 행렬표기(벡터표기), 우측은 성분표기 라고 부른다.

<br>

### 4.2.2 합을 내적으로 계산하기

$\Sigma$는 내적으로도 계산할 수 있다.

ex) 1부터 1000까지의 합

1+2+3+...+1000 = $\begin{bmatrix}1\\1\\\ \vdots\\1\end{bmatrix}$ $\cdot$ $\begin{bmatrix}1\\2\\\ \vdots\\1000\end{bmatrix}$

=> 파이썬에서는 for문을 사용하지 않아도 내적으로 합을 쉽게 계산할 수 있다.

```python
# 리스트 4-2-(1)
import numpy as np
a = np.ones(1000)
b = np.arange(1, 1001)
print(a.dot(b))
====
500500.0
```

<br>

<br>

## 4.3 곱의 기호

$\prod$는 모든 것을 곱하는 기호.
ex) $\prod_{n=a}^b f(n)$ = f(a)$\times$f(a+1)$\times$ $\cdots$ $\times$ f(b)

<br>

<br>

## 4.4 미분

머신러닝은 _함수에서 최소나 최대인 입력을 찾는 문제 (=최적화 문제)_ 이다.
함수의 최소 지점은 기울기가 0이 되는 성질이 있다.

$\therefore$ 함수의 기울기를 '미분'을 통해 도출한다.

하나의 변수만 변수 취급하고, 나머지 변수는 상수 취급해서 미분하는 '편미분'을 사용.

<br>

### 4.5.3 경사를 그림으로 나타내기

```python
# 리스트 4-2-(2)
import numpy as np
import matplotlib.pyplot as plt

## 함수 f1정의
def f1(w0, w1):
  return w0**2 + 2*w0*w1 + 3

## f의 w0에 대한 편미분
def df_dw0(w0, w1):
  return 2*w0 + 2*w1

## f의 w1에 대한 편미분
def df_dw1(w0, w1):
  return 2*w0 + 0*w1

w_range = 2
dw = 0.25
w0 = np.arange(-w_range, w_range+dw, dw)
w1 = np.arange(-w_range, w_range+dw, dw)
                # == w0 = np.linspace(-w_range, w_range, dw) 동일하다.
wn = w0.shape[0]
ww0, ww1 = np.meshgrid(w0, w1)    #격자 모양으로 나눈 w0와 w1을 2차원 배열 ww0, ww1에 저장.

ff = np.zeros((len(w0), len(w1)))
dff_dw0 = np.zeros((len(w0), len(w1)))
dff_dw1 = np.zeros((len(w0), len(w1)))

## ww0와 ww1에 대한 f, 편미분의 값 계산
for i0 in range(wn):    #E
  for i1 in range(wn):
    ff[i1, i0] = f1(w0[i0], w1[i1])
    dff_dw0[i1, i0] = df_dw0(w0[i0], w1[i1])
    dff_dw1[i1, i0] = df_dw1(w0[i0], w1[i1])

# 그래프로 나타낸다.
plt.figure(figsize=(9, 4))  # 전체 그래프 영역 사이즈 9*4로 설정.
plt.subplots_adjust(wspace = 0.3) # subplot 좌우간격 0.3으로 설정.
plt.subplot(1, 2, 1)    # 그래프 1x2로 2개 나타낼것임. 그 중 좌측 그래프.
cont = plt.contour(ww0, ww1, ff, 10, colors='k')    # f의 등고선 표시(x, y, z축값, 10(=등고선10+1개로 나타낼것임), 컬러값)
cont.clabel(fmt='%2.0f', fontsize=8)    # 등고선 수치 유효숫자2개 정수로 나타냄. 폰트사이즈 8
#plt.xticks(), plt.yticks()는 그래프에 x, y값을 어디부터 어디까지 어느 간격으로 나타낼것인지 설정하는 함수.
plt.xticks(range(-w_range, w_range+1, 1))
plt.yticks(range(-w_range, w_range+1, 1))
#plt.xlim(), plt.ylim()은 그래프의 어느 범위부터 어느 범위를 화면상에 나타낼지 설정하는 함수.
plt.xlim(-w_range-0.5, w_range+0.5)
plt.ylim(-w_range-0.5, w_range+0.5)
#x, y축 범례 이름 지정, 폰트사이즈 14
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)

plt.subplot(1, 2, 2)  # 그래프 1x2 중 우측 그래프 설정.
plt.quiver(ww0, ww1, dff_dw0, dff_dw1)    #경사 벡터를 화살표로 표시.
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.xticks(range(-w_range, w_range+1, 1))
plt.yticks(range(-w_range, w_range+1, 1))
plt.xlim(-w_range-0.5, w_range+0.5)
plt.ylim(-w_range-0.5, w_range+0.5)
plt.show()
====
```

화살표는 경사가 높은 방향을 가리키고, 경사가 가파를수록 화살표의 길이가 길다.
화살표를 따라가면, 어느 지점에서 시작하더라도 그래프의 보다 높은 부분으로 진행한다.

<br>

### 4.5.4 다변수의 중첩 함수의 미분
다변수 함수가 중첩되어 있는 경우의 미분은 여러 층으로 된 신경망의 학습 규칙을 도출할 때 사용한다.

### 4.5.5 합과 미분의 교환
합의 기호로 표현된 함수를 미분할 경우가 존재한다.
합을 계산하고 미분하는 것과 각 항의 미분을 계산하고 합을 구하는 것의 결과가 같다.
=> 미분 기호는 합의 기호의 안쪽에 넣어서 먼저 미분을 계산할 수 있다.

<br>

<br>

## 4.6 행렬
행렬을 사용하면 많은 연립 방정식을 하나의 식으로 나타낼 수 있어서 편리하다.

### 4.6.1 행렬이란

행렬을 하나의 변수로 나타낼 때, __A__ 와 같이 굵은 대문자를 사용한다.
행렬의 성분(요소)의 수를 나타낼 때는 [ __A__ ]$_{i,j}$ 와 같이 쓴다.$\_$
이는 행렬 A의 i행 j열 성분을 나타낸다.

### 4.6.2 행렬의 덧셈과 뺄셈

파이썬에서 행렬의 덧셈과 뺄셈을 수행하기 위해서는 numpy 라이브러리를 import 해야한다.

```python
import numpy as np

A = np.array([[1,2,3], [4, 5, 6]])
print(A)

B = np.array([[7, 8, 9], [10, 11, 12]])
print(B)

print(A+B)
print(A-B)
====
[[1 2 3]
 [4 5 6]]
[[ 7  8  9]
 [10 11 12]]
[[ 8 10 12]
 [14 16 18]]
[[-6 -6 -6]
 [-6 -6 -6]]
```

### 4.6.4 행렬의 곱
`A.dot(B)`는 내적에 국한된 연산이 아니라, 행렬 곱을 계산하는 연산이다.

>파이썬에서 행렬을 곱할 때는 계산이 가능하도록 자동으로 행렬의 방향을 조정한다.
ex) [1, 2, 3][4, 5, 6]을 [1, 2, 3][4, 5, 6]$^T$ 으로 자동 조정.

```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print(A*B)    # 각 요소의 곱셈
print(A.dot(B)) # 행렬의 곱셈(자동으로 변환)
print(A/B)    # 각 요소의 나눗셈
====
[ 4 10 18]
32
[0.25 0.4  0.5 ]
```

```python
A = np.array([[1, 2, 3], [-1, -2, -3]])
B = np.array([[4, -4], [5, -5], [6, -6]])
print(A.dot(B))
====
[[ 32 -32]
 [-32  32]]
```

일반적으로 A의 크기가 LxM 행렬이고 B가 MxN 행렬일 때 AB의 크기는 LxN 행렬이다.

알다시피, AB와 BA는 일반적으로 동일하지않다.

<br>

### 4.6.5 단위 행렬
대각선 성분이 1이고 그 이외의 성분은 0인 행렬을 __I__ 로 나타내고, 단위행렬이라 부른다.

파이썬에서는 __`np.identity(n)`__ 명령으로 nxn 단위 행렬을 생성한다.

```python
print(np.identity(3))
====
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

각 요소에는 '.'가 붙어있다. 각 성분이 소수도 나타낼 수 있는 float형임을 나타낸다.

단위행렬은 그 어떤 행렬과 곱해도 결과가 원행렬과 달라지지않는 곱셈의 항등원이다.

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.identity(3)
print(A)
print(A.dot(B))
====
[[1 2 3]
 [4 5 6]
 [7 8 9]]
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

<br>

### 4.6.6 역행렬

행렬의 나누기를 생각한다.
A * 역행렬 = I 가 되는 행렬.
A의 역행렬을 A$^{-1}$로 표시한다.

일반적으로 행렬의 곱셈은 순서에 따라 결과가 달라지지만, 역행렬은 곱하는 순서에 관계없이 결과가 단위행렬이 된다.

파이썬에서 __`np.linalg.inv(A)`__ 로 A의 역행렬을 구할 수 있다.

```python
A = np.array([[1, 2], [3, 4]])
B = np.linalg.inv(A)
print(A)
print(B)
print(A.dot(B))
====
[[1 2]
 [3 4]]
[[-2.   1. ]
 [ 1.5 -0.5]]
[[1.0000000e+00 0.0000000e+00]
 [8.8817842e-16 1.0000000e+00]]
```

역행렬이 존재하지 않는 행렬도 있음에 주의하자.

<br>

### 4.6.7 전치
세로 벡터 <--전치--> 가로 벡터
전치연산은 __T__ 로 나타낼 수 있다.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = A.T
print(A)
print(B)
====
[[1 2 3]
 [4 5 6]]
[[1 4]
 [2 5]
 [3 6]]
```

AB를 한번에 전치하는 경우, 행렬 곱의 순서가 반대가 되어(AB)$^T$ = B$^T$A$^T$가 된다.

<br>

### 4.6.9 행렬과 사상

행렬은 '벡터를 다른 벡터로 변환하는 규칙'으로 해석할 수 있다.
행렬은 선형 사상으로 분류된다.
행렬 [\[2, -1], [1, 1]][x, y] = [\[2x-y], [x+y]] 이 되므로
행렬 [\[2, -1], [1, 1]]은 점 [x, y]를 [2x-y, x+y]로 옮기는 사상으로 해석할 수 있다.

<br>

<br>

## 4.7 지수 함수와 로그 함수

클래스 분류 문제에서 사용하는 시그모이드 함수, 소프트맥스 함수는 exp(x)를 포함하는 지수 함수로 만들어져있다.
이러한 함수를 미분할 필요성이 있다.

머신러닝에서는 매우 큰 수나 매우 작은 수를 처리하게 되는데, 프로그램에서 이러한 수를 취급하는 경우 오버플로를 일으킬 수도 있다. -> 로그를 통해서 오버플로를 방지한다.

로그는 곱셈을 덧셈으로 변환하는 성질을 가지고 있다. -> 곱셈으로 표현되는 확률을 덧셈으로 변환할 수도 있다.
로그는 단조 증가 함수이기 때문에, 최솟값은 바뀌더라도 최솟값을 취하는 값은 변하지 않는다. 최댓값을 찾을 때도 마찬가지. -> 최댓값, 최솟값이 되는 값을 찾을때(오차함수를 미분할 때) 로그함수를 이용한다.

### 4.7.3 지수 함수의 미분

y' = (a^x)' = a^x log a

밑이 e일 경우를 제외하면 log a 만큼 함수의 형태가 변하지만,
__밑이 e인 경우 log e = 1이므로 함수의 형태가 변하지 않는다.__ 따라서 미분 계산을 할 때 편리해지고, e를 밑으로 하는 지수 함수를 다양한 곳에서 사용한다.

<br>

### 4.7.4 로그 함수의 미분

y'(x) = (log x)' = 1/x

<br>

### 4.7.5 시그모이드 함수

시그모이드 함수는 아래와 같이 정의되는, 매끄러운 계단 형태를 한 함수이다.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = 1/(1+np.exp(-x))

plt.figure(figsize=(4, 4))
plt.plot(x, y, 'black', linewidth=3)

plt.ylim(-1, 2)
plt.xlim(-10, 10)
plt.grid(True)
plt.show()
====
```

시그모이드 함수는 __음에서 양의 실수를 0에서 1 사이의 실수로 변환__ 하기 때문에 __확률__ 을 나타낼 때 자주 사용한다.

시그모이드 함수의 미분은
f(x) = 1+exp(-x) 이고, 시그모이드 함수가 1/f(x)의 형태라고 생각하면,
__y' = y(1-y)__ 의 형태로 도출된다.

<br>

### 4.7.6 소프트맥스 함수

3개의 수 x0=2, x1=1, x2=-1이 있고, 이 수의 대소 관계를 유지하면서 각각의 확률을 나타내는 y0, y1, y2로 변환하려고 한다.
확률이므로 y0+y1+y2 = 1이 되어야 한다.
이런 경우에 __소프트맥스 함수(softmax function)__ 를 사용한다.

먼저 각 xi의 exp의 합계 = u를 구해둔다.
u = exp(x0) + exp(x1) + exp(x2)

y0 = exp(x0)/u, y1=exp(x1)/u, y2=exp(x2)/u 가 된다.

```python
def softmax(x0, x1, x2):
  u = np.exp(x0)+np.exp(x1)+np.exp(x2)
  return np.exp(x0)/u, np.exp(x1)/u, np.exp(x2)/u

#test
y = softmax(2, 1, -1)
print(np.round(y, 2))   #계산된 y값을 소수점 2자리로 반올림하여 나타낸다.
print(np.sum(y))
====
[0.71 0.26 0.04]
1.0     # 합이 1이 되는 것을 확인할 수 있다.
```

소프트맥스 함수는 3개의 변수뿐 아니라 그 이상의 개수의 변수에도 사용할 수 있다.

<br>

### 4.7.7 소프트맥스 함수와 시그모이드 함수
2변수일 때 소프트맥수 함수는 아래와 같다.

=> 시그모이드 함수를 다변수로 확장시킨 것이 소프트맥스 함수라고 할 수 있다.

<br>

### 4.7.8 가우스 함수
가우스 함수(Gaussian function)은
y = exp(-x^2)
형태인 함수이다.

x=0을 중심으로 종 모양을 하고 있다.
가우스 함수의 중심(평균)을 $\mu$로 나타내고, 확산되는 크기(표준편차)를 $\sigma$, 높이를 $\alpha$로 하여 조절 가능한 형태로 변형하면 가우스 함수는
y = a exp(-(x-$\mu$)^2 / $\sigma$^2 ) 의 형태가 된다.

```python
def gauss(mu, sigma, a):
  return a*np.exp(-(x-mu)**2 / sigma**2)

x = np.linspace(-4, 4, 100)
plt.figure(figsize=(4, 4))
plt.plot(x, gauss(0, 1, 1), 'black', linewidth=3)   # 평균 0, 표준편차 1, 높이 1인 가우스 함수
plt.plot(x, gauss(2, 3, 0.5), 'red', linewidth=3)   # 평균 2, 표준편차 3, 높이 0.5인 가우스 함수
plt.ylim(-.5, 1,5)
plt.xlim(-4, 4)
plt.grid(True)
plt.show()
====

```

가우스 함수에서 확률 분포를 나타낼 수 있지만, 그런 경우에는 x에 관한 적분이 1이 되도록 a를 1/((2$\pi$$\sigma^2$)$^{1/2}$)로 만든다.

<br>

### 4.7.9 2차원 가우스 함수

가우스 함수를 2차원으로 확장할 수 있다.
입력을 2차원 벡터[x0, x1]^T라고 하면
가우스 함수의 기본 형식은 y = exp(-(x0^2 + x1^2))가 된다.
그래프로 나타내면 원점을 중심으로 동심원을 가지는 형태가 된다.

```python
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import axes3d
%matplotlib inline

#가우스 함수
# x: Nx2의 행렬, mu:크기가 2인 벡터, sigma: 2x2행렬
def gauss(x, mu, sigma):
  N, D = x.shape    # x의 행, 열의 개수 N, D에 저장
  c1 = 1/(2*np.pi)**(D/2)
  c2 = 1/(np.linalg.det(sigma)**(1/2))
  inv_sigma = np.linalg.inv(sigma)  #sigma의 역행렬을 구한다.
  c3 = x-mu
  c4 = np.dot(c3, inv_sigma)
  c5 = np.zeros(N)

  for d in range(D):
    c5 = c5 +c4[:, d]*c3[:, d]
  p = c1*c2*np.exp(-c5/2)
  return p

#test
x = np.array([[1, 2], [2, 1], [3, 4]])
mu = np.array([1, 2])
sigma = np.array([[1, 0], [0, 1]])
print(gauss(x, mu, sigma))
====
[0.15915494 0.05854983 0.00291502]
```
