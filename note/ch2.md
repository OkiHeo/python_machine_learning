## 2.1 사칙 연산
### 2.1.1 사칙 연산의 이용
```python
(1+2*3-4)/5
----
0.6
```
보통 사칙연산 __+, -, *, /__ 과 동일하다.

### 2.1.2 제곱
__**__ 으로나타낸다.
```python
2**8
----
256
```


## 2.2 변수
```python
x=1
y=1/3
x+y
----
1.333333333333
```
변수는 대소문자를 구별하며, 첫 문자에 숫자를 사용할 수 없고, 알파벳, 숫자, 밑줄을 사용할 수 있다.
ex) alpha_a123 (O)
ex) 1_234abc   (X)


## 2.3 자료형
* int : 정수
* float : 실수
* str : 문자열
* bool : 참과 거짓
* list : 배열
* tuple : 배열(수정불가)
* ndarray : 행렬

### 2.3.2 type으로 자료형 알아보기
```python
type(100)
----
int
```
```python
type(10.1)
----
float
```
변수명에 넣는 데이터에 따라 자동으로 변수의 형이 결정된다.

### 2.3.3 문자열
```python
x = 'learning'
type(x)
----
str
```
문자열은 ', ", ''',""" 으로 둘러싸서 만들 수 있다.


## 2.4 print문
```python
x = 1/3
x
y = 2/3
y
----
0.666666666666
```
print문을 사용하지 않고 변수명만 입력하면 셀의 제일 마지막에 있는 y의 값만 출력된다.

### 2.4.2 문자열과 수치를 함께 표시
```python
print('x='+str(x))
----
x=0.333333333333
```
여러 변수를 표시할 경우에는 문자열 내에 {0}, {1}, {2}를 지정한다.  
```python
x=1/3
y=1/7
z=1/4
print('weight: {0} kg {1}kg {2}kg'.format(x, y, z))
----
weight: 0.3333333333333333 kg 0.14285714285714285kg 0.25kg
```
{수치:.nf}를 입력하면 소수점 이하 n자리까지 표시한다. (반올림)
```python
print('weight: {0:.3f}kg {1:.3f}kg {2:.3f}kg'.format(x, y, z))
----
weight: 0.333kg 0.143kg 0.250kg
```


## 2.5 list
### 2.5.1 list의 이용
리스트는 여러 데이터를 하나의 단위로 취급하고 싶은 경우. __배열 변수를 사용하고 싶을 때__ 사용한다.
__리스트명[]__ 으로 나타낸다.
```python
x=[1, 1, 2, 4, 5]
print(x)
----
[1, 1, 2, 4, 5]
```
각 성분은 리스트명[요소번호]로 참조할 수 있다.

여러 자료형을 혼합하여 저장할 수도 있다.
```python
s=['SUN', 1, 'MON', 2]
print(type(s[0]))
print(type(s[1]))
print(s)
----
<class 'str'>
<class 'int'>
['SUN', 1, 'MON', 2]
```

### 2.5.2 2차원 배열
```python
a = [[1, 2, 3], [5, 6, 7]]
print(a)
---
[[1, 2, 3], [5, 6, 7]]
```
개별 요소의 참조는 변수명[i][j]로 한다.
ex) 위 예시에서 5를 10으로 변경하고 싶을 때
`a[1][0] = 10`을 실행하면 된다.

### 2.5.3 list의 길이
len을 사용해서 확인 가능.
```python
x=[1, 2, 3]
print(len(x))
----
3
```

### 2.5.4 연속된 정수 데이터의 작성
5~9의 연속된 정수 데이터를 만들기 위해서
range(시작숫자, 끝숫자+1)을 사용한다.
```python
y=range(5, 10)
print(type(y))
print(list(y))
print(type(list(y)))
----
<class 'range'>
[5, 6, 7, 8, 9]
<class 'list'>
```
list(변수명)으로 range형을 list형으로 변환하여 출력하였다.

0부터 시작하는 수열 데이터를 만들고 싶다면 range(끝숫자+1)을 사용한다.


## 2.6 tuple
tuple은 요소를 수정할 수 없는 배열이다.
`(1, 2, 3)`과 같이 ()를 사용하여 배열을 나타낸다.

list와 같이 변수명[요소번호]를 이용해서 참조 가능하다. (수정은 불가)

```python
a = (1)
print(type(a))
b = (1,)
print(type(b))
----
<class 'int'>
<class 'tuple'>
```
길이가 1인 tuple은 (1,)과 같이 쉼표를 붙여주어야 한다.



## 2.7 if문
```python
x=11
if x>10:
  print('x is ')
  print('        larger than 10.')
else:
  print('x is smaller than 11')
----
x is
        larger than 10.
```
파이썬에서 중괄호를 사용하지 않고, 들여쓰기로 구분한다. 스페이스 바*4를 주로 사용함.

### 2.7.2 비교 연산자
* a==b
* a>b
* a>=b
* a<b
* a<=b
* a!=b
연산의 결과는 모두 bool형이다.


## 2.8 for문
```python
for i in [1, 2, 3]:
  print(i)
----
1
2
3
```
list형 대신 tuple, range형을 사용할 수도 있다.
```python
# range(시작, 끝+1, step사이즈)
for i in range(3, 7, 2):
  print(i)
----
3
5
```

```python
num = [2, 4, 6, 8, 10]
for i in range(len(num)):
  num[i] = num[i]*3
print(num)
----
[6, 12, 18, 24, 30]
```

### 2.8.2 enumerate의 이용
```python
num=[2, 4, 6, 8, 10]
for i, n in enumerate(num):
  num[i] = n*2
print(num)
----
[4, 8, 12, 16, 20]
```
<br>
## 2.9 벡터
list간의 +연산이 벡터 연산처럼 동작할까?
```python
[1, 2] + [3, 4]
----
[1, 2, 3, 4]
```
아니!
str형의 +연산과 마찬가지로 연결된다.

### 2.9.1 numpy의 이용
```python
import numpy as np
```
`as np`부분은 numpy를 np라는 별명을 붙여서 사용하겠다는 의미.
앞으로 `np."function"`으로 numpy관련 함수를 호출할 수 있다.

### 2.9.2 벡터의 정의
앞으로 numpy관련 기능을 사용할 때는 `import numpy as np`가 항상 정의되어있다고 가정한다.
```python
x = np.array([1, 2, 3])
x
----
array([1, 2, 3])
```
```python
# print를 사용해서 벡터를 출력하면 요소 사이의 ,가 생략되어 출력된다.
print(x)
----
[1 2 3]
```

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(x+y)
----
[5 7 9]
```

```python
type(x)
----
numpy.ndarray
```
벡터도 배열과 마찬가지로 하나의 요소를 참조하고, 수정하기 위해서 []를 사용한다.
```python
x[0]=8
print(x)
----
[8 2 3]
```

### 2.9.5 연속된 정수 벡터의 생성
**np.arange(n)** 으로 요소의 값이 1씩 증가하는 벡터 배열을 생성할 수 있다.
```python
print(np.arange(10))
----
[0 1 2 3 4 5 6 7 8 9]
```
```python
print(np.arange(4, 10))
----
[4, 5, 6, 7, 8, 9]
```

+) 사칙연산
```python
# !사이즈가 다른 벡터끼리 연산은 불가능하다.
a = np.arange(10)
b = np.arange(5, 16)
print(a+b)
----
[ 5  7  9 11 13 15 17 19 21 23]
```
<br>
### 2.9.6 ndarray형의 주의점
ndarray형의 내용을 복사하기 위해서는 일반 변수들처럼 a=b형태로 사용하면 안된다.
`a = b.copy()`를 사용해야한다.<br>
**단순히 a=b를 사용하면 b의 내용을 변경하면 변경사항이 a에도 반영된다.**
b의 내용이 저장된 곳의 참조 주소가 전달되기 때문이다.

<br>
## 2.10 행렬
ndarray 2차원 배열로 행렬을 정의할 수 있다.
```python
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)
----
[[1 2 3]
 [4 5 6]]
```

### 2.10.2 행렬의 크기
`ndarray변수명.shape`으로 행렬의 크기를 알 수 있다.
```python
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x.shape)
----
(2, 3)
```

```python
print(type(x.shape))
----
<class 'tuple'>
```
x.shape는 tuple형을 리턴한다.

### 2.10.3 요소의 참조, 수정
```python
x = np.array([[1, 2, 3], [4, 5, 6]])
x[1, 2] = 9
print(x)
----
[[1 2 3]
 [4 5 9]]
```

### 2.10.5 요소가 0과 1인 ndarray만들기
**np.zeros(), np.ones()** 로 만들 수 있다.
```python
print(np.zeros(10))
----
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```
```python
# 2x10의 영행렬 생성
print(np.zeros((2, 10)))
----
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

## 2.10.6 요소가 랜덤인 행렬 생성
**np.random.rand(size)** 를 이용한다.
평균 0, 분산 1의 가우스 분포로 난수를 size개 생성할 수 있다.
```python
np.random.rand(20)
----
array([0.63444793, 0.59376726, 0.68156781, 0.52048412, 0.94561051,
       0.82415997, 0.16225999, 0.19084766, 0.02568032, 0.19530375,
       0.0580724 , 0.02686245, 0.017644  , 0.56497843, 0.52536116,
       0.42502817, 0.69618506, 0.10699665, 0.33042691, 0.80630982])
```
1x20의 난수 행렬을 생성할 수 있다.
```python
np.random.rand(2, 3)
----
array([[0.74380303, 0.29029885, 0.35658039],
       [0.25869059, 0.53869806, 0.83441483]])
```

난수는 0에서 1사이의 균일한 분포를 보인다.<br>
**np.random.randint(low, high, size)** 를 사용하여 low~high의 임의의 정수값으로 이뤄진 size크기의 행렬을 생성할 수 있다.
```python
np.random.randint(1, 10, 5)
----
array([1, 7, 8, 5, 2])
```
난수의 값은 실행할 때마다 달라진다.

### 2.10.7 행렬의 크기 변경
**변수명.reshape(n, m)** 을 사용.
n * m 사이즈의 행렬로 크기를 변경해준다.
```python
a = np.arange(10)
a = a.reshape(2, 5).copy()
print(a)
----
[[0 1 2 3 4]
 [5 6 7 8 9]]
```
<br>
## 2.11 행렬(ndarray)의 사칙연산
```python
x = np.array([[4, 4, 4], [8, 8, 8]])
y = np.array([[1, 1, 1], [2, 2, 2]])
print('x+y\n', x+y)
print('x-y\n', x-y)
print('x*y\n', x*y)
print('x/y\n', x/y)
----
x+y
 [[ 5  5  5]
 [10 10 10]]
x-y
 [[3 3 3]
 [6 6 6]]
x*y
 [[ 4  4  4]
 [16 16 16]]
x/y
 [[4. 4. 4.]
 [4. 4. 4.]]
```
### 2.11.2 스칼라 x 행렬
```python
x = np.arange(5)
print(2*x)
----
[0 2 4 6 8]
```

### 2.11.3 산술 함수
* 모든 요소에 적용되는 함수
  * np.exp(x) : 밑이 e(자연상수)인 지수함수
  * np.sqrt(x) : 제곱근
  * np.log(x) : 밑이 e인 로그
  * np.round(x, 유효자릿수) : 반올림
* 모든 요소에 대해 하나의 값을 반환하는 함수
  * np.mean(x) : 평균
  * np.std(x) : 표준 편차
  * np.max(x) : 최댓값
  * np.min(x) : 최솟값

### 2.11.4 행렬 곱의 계산
```python
v = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1, 1], [2, 2], [3, 3]])
print(v)        # 2*3행렬
print(w)        # 3*2행렬
print(v.dot(w)) # 2*2행렬
----
[[1 2 3]
 [4 5 6]]
[[1 1]
 [2 2]
 [3 3]]
[[14 14]
 [32 32]]
```

## 2.12 슬라이싱
list와 ndarray에서 요소를 한 번에 나타낼 때 슬라이스라는 방법을 사용하면 편리하다.
**':'** 를 사용하여 나타내고,
**변수명[:n]** 와 같이 사용해서 요소번호 0~n-1까지를 한 번에 볼 수 있다.
변수명[:] -> 전체 내용
변수명[n: m: k] -> n~m-1을 k간격으로
```python
x = np.arange(10)
print(x)
print(x[:5])
print(x[::2])     # step사이즈 2씩 건너뛰며 나타내줌
----
[0 1 2 3 4 5 6 7 8 9]
[0 1 2 3 4]
[0 2 4 6 8]
```
<br>
## 2.13 조건을 만족하는 데이터의 수정
### 2.13.1 bool 배열 사용
numpy를 통해 행렬 데이터에서 특정 조건을 만족하는 것을 추출 하여 쉽게 수정할 수 있다.
```python
x = np.array([1, 1, 2, 3, 5, 8, 13])
print(x>3)
----
[False False False False  True  True  True]
```
아래와 같이 활용할 수 있다.
```python
print(x)      # 변경 전
x[x>3] *=10
print(x)      # 변경 후
----
[ 1  1  2  3  5  8 13]
[  1   1   2   3  50  80 130]
```

<br>
## 2.14 Help의 사용
함수의 설명을 확인하기 위해 **help(함수명)** 을 사용할 수 있다.
함수를 어떻게 사용하는지, 파라미터는 무엇인지, 사용 예시를 보여준다.
```python
help(np.random.randint)
----
Help on built-in function randint:

randint(...) method of numpy.random.mtrand.RandomState instance
    randint(low, high=None, size=None, dtype='l')

    Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int or array-like of ints, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
        If array-like, must contain integer values
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.int'.

        .. versionadded:: 1.11.0

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    random.random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted.

    Examples
    --------
    >>> np.random.randint(2, size=10)
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
    >>> np.random.randint(1, size=10)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Generate a 2 x 4 array of ints between 0 and 4, inclusive:

    >>> np.random.randint(5, size=(2, 4))
    array([[4, 0, 2, 1], # random
           [3, 2, 2, 0]])

    Generate a 1 x 3 array with 3 different upper bounds

    >>> np.random.randint(1, [3, 5, 10])
    array([2, 2, 9]) # random

    Generate a 1 by 3 array with 3 different lower bounds

    >>> np.random.randint([1, 5, 7], 10)
    array([9, 8, 7]) # random

    Generate a 2 by 4 array using broadcasting with dtype of uint8

    >>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
    array([[ 8,  6,  9,  7], # random
           [ 1, 16,  9, 12]], dtype=uint8)
```

## 2.15 함수
자주 반복되어 사용되는 코드는 함수로 정의해서 사용하는 것이 좋다.
**def 함수명():** 으로 시작, 함수의 내용은 들여쓰기로 정의한다.
실행을 위해서는 **함수명()** 을 입력하면 된다.
```python
def my_func(x):
  print(x+10)
  print(x*2)
  return x

print('주어진 값: ', my_func(5))
----
15
10
주어진 값: 5
```

어떤 형태로든 인수나 반환값을 만들 수 있고, 여러 개의 반환값을 정의할 수도 있다.
* EXAMPLE1
  어떤 데이터를 1차원 ndarray형으로 함수에 전달해서 평균값과 표준편차를 출력하는 함수
  ```python
  def my_func1(d):
    m = np.mean(d)
    s = np.std(d)
    return m, s

  # 실행부
  data = np.random.randint(100)
  data_mean, data_std = my_func1(data)
  print('mean: {0:3.2f}, std: {1:3.2f}'.format(data_mean, data_std))
  ----
  mean: 20.00, std: 0.00
  ```

* EXAMPLE2
반환값이 여러개라도 하나의 변수로 받을 수 있다.
아래의 예시는 위의 data와 같은 data를 사용한다.
  ```python
  result = my_func1(data)
  print(result)
  print(type(result))
  print('mean: {0:3.2f}, std: {1:3.2f}'.format(result[0], result[1]))
  ----
  (20.0, 0.0)
  <class 'tuple'>
  mean: 20.00, std: 0.00
  ```
  반환값이 여러 개인 함수의 결과를 하나의 변수에 저장하면 tuple형으로 저장된다.
<br>
## 2.16 파일 저장
### 2.16.1 하나의 ndarray형을 저장
하나의 ndarray형을 파일에 저장하려면 **np.save('파일명.npy', 변수명)** 을 사용하면 된다.
데이터를 파일로부터 읽기 위해서는 **np.load('파일명.npy')** 를 사용한다.
```python
data = np.random.rand(10)
print(data)
np.save('data.npy', data)   # data 내용 파일로 저장
data= []    # data변수에 저장되어있던 데이터를 삭제한다.
print(data)
data = np.load('data.npy')
print(data)
----
[0.78649595 0.20022429 0.38863239 0.44387385 0.56229891 0.76832141
 0.00098621 0.84624038 0.12642187 0.263973  ]
[]
[0.78649595 0.20022429 0.38863239 0.44387385 0.56229891 0.76832141
 0.00098621 0.84624038 0.12642187 0.263973  ]
```

### 2.16.2 여러 ndarray형을 저장
**np.savez('파일명.npz', 변수명1newname=변수명1, 변수명2newname=변수명2, ...)** 를 사용한다.
파일로부터 데이터 읽어올 때는 **np.load**를 그대로 사용한다.
* outfile에 np.load를 저장한다면 __outfile['변수명']__ 으로 각각의 변수를 참조할 수 있다.
__outfile.files__ 로 저장된 변수의 목록을 볼 수 있다.
```python
data1 = np.array([1, 2, 3])
data2 = np.array([10, 20, 30])
print(data1, data2)
np.savez('dataz.npz', d1 = data1, d2 = data2)   # 데이터를 파일에 저장
data1 = []
data2 = []
print(data1, data2)
outfile = np.load('dataz.npz')     # 저장한 데이터를 load
print(outfile.files)
data1 = outfile['d1']
data2 = outfile['d2']
print(data1, data2)
----
[1 2 3] [10 20 30]
[] []
['d1', 'd2']
[1 2 3] [10 20 30]
```
