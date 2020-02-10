## 3.1 2차원 그래프 그리기
### 3.1.1 임의의 그래프 그리기
**matplotlib** 의 **pyplot** 라이브러리를 import하고 plt라는 별명을 만들어 사용한다.
주피터 노트북에서 그래프를 표시하기 위해 `%matplotlib inline`명령을 추가한다.
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#data 작성
np.random.seed(1)     #난수를 고정하기 위해 시드를 1로 준다.
x = np.arange(10)     # 0~9의 값을 가지는 1x10행렬 x에 저장
y = np.random.rand(10)

#그래프 표시
plt.plot(x, y)    #꺾은선그래프를 등록한다. x와 y를 축으로 사용.
plt.show()        #그래프를 출력한다.
====

```

### 3.1.2 프로그램 리스트 규칙
프로그램 리스트 번호의 규칙을 정해보자.
리스트 번호는 1-(1), 1-(2), 1-(3), 2-(1), ... 와 같이 사용한다.
괄호 앞의 숫자가 같은 리스트는 변수를 공유하는 리스트이다.
괄호 안의 숫자 순서대로 실행한다고 약속하자.
만약에 2-(1)과 같이 괄호 앞의 숫자가 바뀌면 1-(x)는 일절 사용하지 않는다고 약속하자.

### 3.1.3 3차 함수 f(x)=(x-2)x(x+2) 그리기
먼저 함수 f(x)를 정의한다.
```python
#리스트 2-(1)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def f(x):
  return (x-2)*x*(x+2)
```

```python
#리스트 2-(2)
print(f(1))
====
-3
```

x를 ndarray배열로 지정하고 함수 f의 파라미터로 넣어보자.
```python
#리스트 2-(3)
x = np.array([1, 2, 3])
print(f(x))
====
[-3 0 15]
```
벡터의 사칙연산이 각 요소마다 이루어진다.
<br>
### 3.1.4 그리는 범위 결정하기
```python
#리스트 2-(4)
x = np.arange(-3, 3.5, 0.5)
print(x)
====
[-3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2.   2.5  3. ]
```
동일하게, `linspace`라는 명령을 사용하는 방법도 있다.
`linspace(start, end, n)`은 start에서 end사이를 일정 간격으로 n개로 나눈 값을 리턴해준다.__(end를 포함!)__
```python
#리스트 2-(5)
x = np.linspace(-3, 3, 10)
print(np.round(x, 2))
====
[-3.   -2.33 -1.67 -1.   -0.33  0.33  1.    1.67  2.33  3.  ]
```
<br>

### 3.1.5 그래프 그리기
정의역 x, 치역 f(x)인 그래프를 간단히 그려낼 수 있다.

```python
#리스트 2-(6)
plt.plot(x, f(x))
plt.show()
====
```

### 3.1.6 그래프를 장식하기
그래프 내부에 눈금, 축의 이름, 범례를 넣어보자.
```python
#리스트 2-(7)
#그래프로 그릴 함수를 정의한다.
def f2(x, w):
  return (x-w)*x*(x+2)

#x를 정의한다.
x = np.linspace(-3, 3, 100) #-3~3을 100분할

#그래프 묘사
plt.plot(x, f2(x, 2), color='black', label='$w=2$')
plt.plot(x, f2(x, 1), color='green', label='$w=1$')
plt.legend(loc="upper left")  #범례표시 좌상단에.
plt.ylim(-15, 15)             #y축의 범위
plt.title('$f_2(x)$')         #제목
plt.xlabel('$x$')             #x축 라벨
plt.ylabel('$y$')             #y축 라벨
plt.grid(True)                #그리드 표시
plt.show()
====
```

그래프의 색상은 `plt.plot(x, f2(x, 2), color='_____')` 으로 정할 수 있다.
ex) color=(255, 0, 0) 으로 RGB값을 지정할 수도 있다.

범례는 `plt.plot(x, f2(x, 2), label='_____')`로 지정하고 `plt.legend(loc="_____")`로 나타낸다.
loc은 "upper right", "upper left", "lower left", "lower right"로 지정할 수 있다.

문자열은 __$__ 로 TeX기반의 수식으로 지정할 수 있다.

### 3.1.7 그래프를 여러 개 보여주기
여러 개의 그래프를 나란히 표시하기 위해서는 `plt.subplot(n1, n2, n)`을 사용한다.
전체를 세로 n1, 가로 n2로 나눈 n번째에 그래프가 그려진다. _n은 1에서 시작한다._
```python
#리스트 2-(9)
plt.figure(figsize=(10, 3))                     #전체 영역의 크기 지정
plt.subplots_adjust(wspace=0.5, hspace=1)     #그래프의 간격을 설정한다.

#그래프 6개를 나란히 표시한다.
for i in range(6):
  plt.subplot(2, 3, i+1)    #그래프 위치를 지정한다.
  plt.title(i+1)
  plt.plot(x, f2(x, i), 'k')
  plt.ylim(-20, 20)
  plt.grid(True)
plt.show()
====
```

`plt.figure(figsize=(w, h))`는 전체 영역의 크기를 지정한다.
subplot 내부의 상하좌우간격은 `plt.subplots_adjust(wspace=w, hspace=h)`로 조절할 수 있다.
<br>
## 3.2 3차원 그래프 그리기
```python
#리스트 3-(1)
import numpy as np
import matplotlib.pyplot as plt라는

#함수3 정의
def f3(x0, x1):
  r = 2* x0**2 + x1**2
  ans = r*np.exp(-r)
  return ans

#x0, x1에서 f3을 각각 계산한다.
xn = 9
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)
y = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
  for i1 in range(xn):
    y[i1, i0] = f3(x0[i0], x1[i1])
```
x0과 x1은 9개의 요소로 구성된 행렬이다.
```python
#리스트 3-(2)
print(x0)
====
[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
```
numpy함수 `np.round(y, n)`을 이용해서 y의 값을 소수점 n자리로 반올림해서 정리해서 출력한다.
```python
#리스트 3-(3)
print(np.round(y, 1))
====
[[0.  0.  0.  0.  0.1 0.  0.  0.  0. ]
 [0.  0.  0.1 0.2 0.2 0.2 0.1 0.  0. ]
 [0.  0.  0.1 0.3 0.4 0.3 0.1 0.  0. ]
 [0.  0.  0.2 0.4 0.2 0.4 0.2 0.  0. ]
 [0.  0.  0.3 0.3 0.  0.3 0.3 0.  0. ]
 [0.  0.  0.2 0.4 0.2 0.4 0.2 0.  0. ]
 [0.  0.  0.1 0.3 0.4 0.3 0.1 0.  0. ]
 [0.  0.  0.1 0.2 0.2 0.2 0.1 0.  0. ]
 [0.  0.  0.  0.  0.1 0.  0.  0.  0. ]]
```
중심과 주변이 0인 도너츠 모양을 하고 있지만, 눈으로 쉽게 확인할 수 없다.

### 3.2.2 수치를 색으로 표현하기
`plt.pcolor(2차원ndarray)`를 사용해서 2차원 행렬의 요소를 색상으로 표현한다.
```python
#리스트 3-(4)
plt.figure(figsize=(3.5, 3))
plt.gray()        #색상을 회색 음영으로 표시한다.
plt.pcolor(y)     #행렬을 색상으로 표현한다.
plt.colorbar()    #행렬 옆에 컬러 바를 나타낸다.
plt.show()
====

```

### 3.2.3 함수의 표면을 표시: surface
3차원의 입체 그래프로 표현해보자!
```python
#리스트 3-(5)
from mpl_toolkits.mplot3d import Axes3D

xx0, xx1 = np.meshgrid(x0, x1)              #좌표점 x0, x1으로 xx0, xx1을 만든다.

plt.figure(figsize=(5, 3.5))
ax = plt.subplot(1, 1, 1, projection='3d')  #차트를 3차원으로 만들기 위해 projection을 3D로 설정

#rstride, cstride로 가로, 세로로 몇 개의 선을 긋는지 설정
#alpha로 면의 투명도 설정
ax.plot_surface(xx0, xx1, y, rstride=1, cstride=1, alpha=0.3, color='blue', edgecolor='red')
ax.set_zticks((0, 0.2))       #z의 눈금을 0과 0.2로 제한
ax.view_init(75, -95)         #3차원 그래프의 방향 조절. (상하회전각도, 좌우회전각도)
plt.show()
====

```
`from 라이브러리명 import 함수명`으로 라이브러리를 import하면 `라이브러리명.함수명`이 아니라 `함수명`만으로도 함수를 호출할 수 있다.

`np.meshgrid(x0, x1)`으로 생성된 xx0와 xx1은 아래와 같이 나타난다.
```python
#리스트 3-(6)
print(x0)
print(x1)
====
[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
```

```python
#리스트 3-(7)
print(xx0)        #x0를 가로 행렬으로 ↑방향으로 쌓은 것
print(xx1)        #x1을 세로 행렬으로 →방향으로 쌓은 것
====
[[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]]
[[-2.  -2.  -2.  -2.  -2.  -2.  -2.  -2.  -2. ]
 [-1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5]
 [-1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. ]
 [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5]
 [ 1.   1.   1.   1.   1.   1.   1.   1.   1. ]
 [ 1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5]
 [ 2.   2.   2.   2.   2.   2.   2.   2.   2. ]]
```
`ax.plot_surface(xx0, xx1, y, rstride=1, cstride=1, alpha=0.3, color='blue', edgecolor='red')`
rstride와 cstride값은 자연수여야하고, 이 값이 작을수록 선의 간격이 조밀해진다.

z축의 눈금은 기본 상태로 표시하면 숫자가 겹치기 때문에 `ax.set_zticks((0, 0.2))`로 z의 눈금을 0과 0.2로 제한했다.

### 3.2.4 등고선으로 표시: contour
함수의 높이를 알아보기 위해서는 등고선 플롯이 편리하다.
<br>
해상도를 50x50로해서 xx0, xx1, y를 생성한다. __contour플롯은 해상도를 어느 정도 높이지 않으면 정확하게 표시되지 않는다.__
```python
#리스트 3-(9)
xn = 50
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)

y = np.zeros((len(x0), len(x1)))
for i0 in range(xn):
  for i1 in range(xn):
    y[i1, i0] = f3(x0[i0], x1[i1])

xx0, xx1 = np.meshgrid(x0, x1)

plt.figure(1, figsize=(4, 4))
cont = plt.contour(xx0, xx1, y, 5, colors='black')
cont.clabel(fmt='%3.2f', fontsize=8)
plt.xlabel('$x_0$', fontsize=14)
plt.ylabel('$x_1$', fontsize=14)
plt.show()
====

```
`plt.contour(xx0, xx1, y, 5, colors='black')`으로 등고선 플롯을 작성한다.
`5`는 표시하는 높이를 5단계로 구분하겠다는 의미이다.
`plt.contour`의 반환값을 cont에 저장하고, `cont.clabel(fmp='%3.2f', fontsize=8)`명령으로 각 등고선에 숫자를 넣을 수 있다.(유효숫자 3개, 소수점 아래 2자리까지 표시하고 fontsize는 8으로 함.)
