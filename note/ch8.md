## 8.1 MNIST 데이터베이스

[The MNIST DATABASE of handwritten](http://yann.levun.com/exdb/mnist/)에서 무료로 다운할 수 있지만 케라스 코드를 이용하면 더 쉽게 이용할 수 있다.

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load.data()
```

60000개의 훈련용 데이터(이미지와 라벨)가 x_train, y_train에 저장되고
10000개의 테스트용 데이터가 x_test, y_test에 저장된다.

__x_train__ 은 60000x28x28 의 배열 변수. i번째 이미지는 x_train[i,:,:] 명령으로 꺼낼 수 있다.
__y_train__ 은 길이가 60000인 1차원 배열 변수. y[i]에는 이미지 i에 대응하는 0-9의 값이 포함되어있다.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(1, figsize=(12, 3.2))
plt.subplots_adjust(wspace=0.5)
plt.gray()
for id in range(3):
    plt.subplot(1, 3, id + 1)
    img = x_train[id, :, :]
    plt.pcolor(255 - img)
    plt.text(24.5, 26, "%d" % y_train[id],
             color='cornflowerblue', fontsize=18)
    plt.xlim(0, 27)
    plt.ylim(27, 0)
    plt.grid('on', color='white')
plt.show()
```

@@@@@@@@@@@@@@@ 이미지

<br>

## 8.2 2층 피드 포워드 네트워크 모델

2층 피드 포워드 네트워크 모델을 사용해서 이 필기체 숫자의 클래스 분류 문제를 해결할 수 있는지 살펴본다.

데이터를 사용하기 쉬운 형태로 변경한다.

```python
from keras.utils import np_utils


x_train = x_train.reshape(60000, 784) # (A)
x_train = x_train.astype('float32') # (B)
x_train = x_train / 255 # (C)
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes) # (D)


x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = np_utils.to_categorical(y_test, num_classes)
```

(A): __x_train__ 의 28x28부분을 784길이의 벡터로 변경한다.
(B): 입력을 실수로 처리하기 위해 int를 float형으로 변환.
(C): 255로 나누어 0\~1의 실수로 변환한다.

(D): __y_train__ 의 요소는 0-9의 정수이며, `np.utils.to_categorical()`이라는 케라스 함수를 사용해서 1-of-K부호화법으로 변경한다.

y_test도 x_test와 동일하게 28x28 -> 784로 변환.

입력은 784차원의 벡터.
10개의 숫자를 분류할 수 있도록 네트워크의 __출력층은 10개의 뉴런으로__ 하고, 각 뉴런의 출력값이 확률을 나타낼 수 있도록 __소프트맥스 함수를 활성화함수로__ 사용.
입력과 출력을 연결하는 __중간층의 뉴런은 16개로__ 하고, __활성화 함수는 시그모이드 함수로__ 한다.

```python
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


model = Sequential() # (A)
model.add(Dense(16, input_dim=784, activation='sigmoid')) # (B)
model.add(Dense(10, activation='softmax')) # (C)
model.compile(loss='categorical_crossentropy',
optimizer=Adam(), metrics=['accuracy']) # (D)
```

(A): __model__ 을 __`Sequential()`__ 으로 정의.
(B): 784차원 입력을 갖는 16개의 중간층
(C): 10개의 출력층 정의
(D): __`model.compile()`__ 의 인수에 `optimizer=Adam()`을 통해 알고리즘을 __Adam__ 으로 결정.(조금 더 발전한 경사하강법이라고 볼 수 있다.)

```python
import time


startTime = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                    verbose=1, validation_data=(x_test, y_test)) # (A)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))
====
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 1s 25us/step - loss: 2.0663 - acc: 0.2861 - val_loss: 1.7901 - val_acc: 0.4952
Epoch 2/10
60000/60000 [==============================] - 1s 22us/step - loss: 1.6092 - acc: 0.6483 - val_loss: 1.4405 - val_acc: 0.7655
Epoch 3/10
60000/60000 [==============================] - 1s 23us/step - loss: 1.3224 - acc: 0.7872 - val_loss: 1.2012 - val_acc: 0.8196
Epoch 4/10
60000/60000 [==============================] - 1s 23us/step - loss: 1.1159 - acc: 0.8241 - val_loss: 1.0217 - val_acc: 0.8443
Epoch 5/10
60000/60000 [==============================] - 1s 21us/step - loss: 0.9570 - acc: 0.8433 - val_loss: 0.8818 - val_acc: 0.8580
Epoch 6/10
60000/60000 [==============================] - 1s 23us/step - loss: 0.8330 - acc: 0.8583 - val_loss: 0.7737 - val_acc: 0.8707
Epoch 7/10
60000/60000 [==============================] - 1s 21us/step - loss: 0.7375 - acc: 0.8690 - val_loss: 0.6899 - val_acc: 0.8794
Epoch 8/10
60000/60000 [==============================] - 1s 20us/step - loss: 0.6629 - acc: 0.8775 - val_loss: 0.6247 - val_acc: 0.8866
Epoch 9/10
60000/60000 [==============================] - 1s 19us/step - loss: 0.6035 - acc: 0.8833 - val_loss: 0.5723 - val_acc: 0.8918
Epoch 10/10
60000/60000 [==============================] - 1s 18us/step - loss: 0.5555 - acc: 0.8890 - val_loss: 0.5296 - val_acc: 0.8952
Test loss: 0.5295710373401642
Test accuracy: 0.8952
Computation time:13.536 sec
```

오차 함수의 기울기 1회 갱신에 사용되는 데이터셋의 크기를 `batch_size=1000`에서 지정하고있다.

<br>

기존의 경사 하강법은 부분 극소값에 도달하면 그곳이 아무리 얕아도 빠져나갈 수 없다.
__확률적 경사 하강법의 경우에는 '휘청거리는 효과 덕분에 국소해를 벗어날 수 있는' 유용한 성질이 있다.__

* (A): `epochs`: 학습 갱신 횟수를 정하는 매개변수
  * ex) 훈련 데이터 60000개, batch_size=1000이면 한 번 학습에 60회의 매개 변수 갱신이 진행된다. epochs=10이면 그 10배인 600회의 매개 변수 갱신이 한 번 학습에 이루어진다.

  : `verbose=1`로 설정하여 매 시기의 학습 평가치가 표시됨.
  최종적으로 테스트 데이터의 상호 엔트로피 오차(Test loss), 정답률(Test accuracy), 계산 시간(Computation time)이 표시된다.

오버 피팅이 일어나지 않았는지 확인하기 위해, 테스트 데이터 오차의 시간 변화를 살펴보자.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


plt.figure(1, figsize=(10, 4))
plt.subplots_adjust(wspace=0.5)


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training', color='black')
plt.plot(history.history['val_loss'], label='test',
         color='cornflowerblue')
plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')


plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='training', color='black')
plt.plot(history.history['val_acc'],label='test', color='cornflowerblue')
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
```

@@@@@@@@ 이미지

정답률이 증가 중이고,
테스트 데이터의 오차가 단조감소하고있으므로 오버 피팅은 일어나지 않았다고 판단할 수 있다.

89.51%의 정확도가 좋은편인지 아닌지 실제 테스트 데이터를 입력했을 때 모델의 출력을 살펴보자.

```python
def show_prediction():
    n_show = 96
    y = model.predict(x_test) # (A)
    plt.figure(2, figsize=(12, 8))
    plt.gray()
    for i in range(n_show):
        plt.subplot(8, 12, i + 1)
        x = x_test[i, :]
        x = x.reshape(28, 28)
        plt.pcolor(1 - x)
        wk = y[i, :]
        prediction = np.argmax(wk)
        plt.text(22, 25.5, "%d" % prediction, fontsize=12)
        if prediction != np.argmax(y_test[i, :]):
            plt.plot([0, 27], [1, 1], color='cornflowerblue', linewidth=5)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")
#-- 메인
show_prediction()
plt.show()
```

@@@@@@@@@@@ 이미지

* (A): y=model.predict(x_test)에서 x_test전체에 대한 모델의 출력 y를 얻을 수 있다.

모델의 성능을 직접 보는 것은 매우 중요하다. 잘못된 학습을 한 경우를 찾아낼 수 있기 때문이다.

<br>

<br>

## 8.3 ReLU 활성화 함수

전통적으로 시그모이드 함수가 활성화 함수로 선호되었지만 최근에는 __ReLU__ 라는 활성화 함수가 인기있다.

시그모이드 함수는 입력 x가 어느 정도 커지면 항상 1에 가까운 값을 출력하기 때문에 입력의 변화가 출력에 반영되기 어렵다.
-> 오차 함수의 가중치 매개 변수에 대한 편미분이 0에 가까운 값이 되어, 경사 하강법의 학습이 늦어지는 문제점이 있다.
=> ReLU를 사용하면 x가 긍정적일 때, 문제점을 해결할 수 있다.

네트워크 중간층의 활성화 함수를 sigmoid에서 ReLU로 바꿔 실행해보면 아래와 같은 결과를 볼 수 있다.

```python
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


model = Sequential()
model.add(Dense(16, input_dim=784, activation='relu')) # (A)
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])


startTime = time.time()
history = model.fit(x_train, y_train, batch_size=1000, epochs=10,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))
====
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 1s 21us/step - loss: 1.5461 - acc: 0.5425 - val_loss: 0.9006 - val_acc: 0.8073
Epoch 2/10
60000/60000 [==============================] - 1s 18us/step - loss: 0.6691 - acc: 0.8450 - val_loss: 0.4987 - val_acc: 0.8772
Epoch 3/10
60000/60000 [==============================] - 1s 18us/step - loss: 0.4513 - acc: 0.8828 - val_loss: 0.3898 - val_acc: 0.9001
Epoch 4/10
60000/60000 [==============================] - 1s 16us/step - loss: 0.3751 - acc: 0.8988 - val_loss: 0.3395 - val_acc: 0.9106
Epoch 5/10
60000/60000 [==============================] - 1s 15us/step - loss: 0.3351 - acc: 0.9076 - val_loss: 0.3103 - val_acc: 0.9169
Epoch 6/10
60000/60000 [==============================] - 1s 18us/step - loss: 0.3092 - acc: 0.9132 - val_loss: 0.2918 - val_acc: 0.9202
Epoch 7/10
60000/60000 [==============================] - 1s 16us/step - loss: 0.2911 - acc: 0.9178 - val_loss: 0.2761 - val_acc: 0.9224
Epoch 8/10
60000/60000 [==============================] - 1s 14us/step - loss: 0.2773 - acc: 0.9217 - val_loss: 0.2670 - val_acc: 0.9257
Epoch 9/10
60000/60000 [==============================] - 1s 15us/step - loss: 0.2664 - acc: 0.9244 - val_loss: 0.2586 - val_acc: 0.9269
Epoch 10/10
60000/60000 [==============================] - 1s 15us/step - loss: 0.2574 - acc: 0.9269 - val_loss: 0.2524 - val_acc: 0.9299
Test loss: 0.25239113925397394
Test accuracy: 0.9299
Computation time:10.334 sec
```

Sigmoid 함수를 사용했을 때의 정확도 89.51%에서 ReLU를 사용했더니 92.92%로 3%p 상승했다.

`show_prediction()`를 실행하면 테스트 데이터 인식의 예를 볼 수 있다.

```python
show_prediction()
plt.show()
```

@@@@@@@@@@@@@ 이미지

이미지로 결과를 확인해보면 3%p의 성능 향상은 큰 개선임을 체감할 수 있다.( 파란색으로 라벨링 된 결과 오류가 줄어들었음 )

<br>

네트워크 모델의 중간층 가중치 매개 변수 : model.layers[0].get_weights()[0]
바이어스 매개 변수 : model.layers[0].get_weights()[1]
출력층의 매개 변수 : model.layers[1].get_weights()[0]

중간층 가중치 매개변수를 그림으로 나타내보자.

```python
# 1층째의 무게 시각화
w = model.layers[0].get_weights()[0]
plt.figure(1, figsize=(12, 3))
plt.gray()
plt.subplots_adjust(wspace=0.35, hspace=0.5)
for i in range(16):
    plt.subplot(2, 8, i + 1)
    w1 = w[:, i]
    w1 = w1.reshape(28, 28)
    plt.pcolor(-w1)
    plt.xlim(0, 27)
    plt.ylim(27, 0)
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("%d" % i)
plt.show()
```

@@@@@@@@@@@@@@@@ 이미지

가중치 값이 양수이면 검은색으로, 음수인 경우에는 흰색으로 표시한다.
검은 부분에 문자 일부분이 있으면 그 뉴런은 활성화하고, 흰 부분에 문자 일부분이 있으면 억제된다.

이 모델의 성능을 개선하기 위해는 __'입력은 2차원 이미지'__ 라는 공간 정보를 사용해야한다.

현재 상태에서, 이미지가 섞여도 학습의 성능은 변하지 않는다.
왜냐하면, 네트워크의 구조가 전결합형이며 모든 입력 성분은 대등한 관계이기 때문이다.

<br>

<br>

## 8.4 공간 필터

__'공간 필터'__ 라는 이미지 처리법을 사용하자.

필터는 2차원 행렬로 표시된다.
이미지의 일부분과 필터 요소를 곱한 합을 이미지를 슬라이드 시키면서 이미지의 전 영역에서 구한다. (__합성곱 연산__ 수행)

원본 이미지의 위치(i, j)의 픽셀 값을 x(i, j), 3x3의 필터를 h(i, j)라고 하면, 합성곱 연산에서 얻어지는 값 g(i, j)는 아래와 같다.

@@@@@@@@@ 필기

필터의 크기는 중심을 결정할 수 있는 홀수 너비가 사용하기 쉽다.

이미지 처리법을 사용하기 위해 이번에는 60000x28x28의 이미지를 그대로 사용하자.

```python
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
```

가로 및 세로 엣지를 강조하는 2개의 필터를 훈련 데이터의 2번째 데이터('4')에 적용해보자.
필터는 `myfil1`과 `myfil2`로 정의한다.

```python
import matplotlib.pyplot as plt
%matplotlib inline


id_img = 2
myfil1 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [-2, -2, -2]], dtype=float) # (A) 가로 엣지를 추출하는 필터
myfil2 = np.array([[-2, 1, 1],
                   [-2, 1, 1],
                   [-2, 1, 1]], dtype=float) # (B) 세로 엣지를 추출하는 필터


x_img = x_train[id_img, :, :, 0]
img_h = 28
img_w = 28
x_img = x_img.reshape(img_h, img_w)
out_img1 = np.zeros_like(x_img)
out_img2 = np.zeros_like(x_img)

# 필터 처리
for ih in range(img_h - 3):
    for iw in range(img_w - 3):
        img_part = x_img[ih:ih + 3, iw:iw + 3]
        out_img1[ih + 1, iw + 1] = \
            np.dot(img_part.reshape(-1), myfil1.reshape(-1))
        out_img2[ih + 1, iw + 1] = \
            np.dot(img_part.reshape(-1), myfil2.reshape(-1))


# - 표시
plt.figure(1, figsize=(12, 3.2))
plt.subplots_adjust(wspace=0.5)
plt.gray()
plt.subplot(1, 3, 1)
plt.pcolor(1 - x_img)
plt.xlim(-1, 29)
plt.ylim(29, -1)
plt.subplot(1, 3, 2)
plt.pcolor(-out_img1)
plt.xlim(-1, 29)
plt.ylim(29, -1)
plt.subplot(1, 3, 3)
plt.pcolor(-out_img2)
plt.xlim(-1, 29)
plt.ylim(29, -1)
plt.show()
```

@@@@@@@@@@@@@@@ 이미지

필터의 수치를 바꾸는 것만으로 대각선 엣지 강조, 이미지 스무딩, 세부 부분의 강조가 가능하다.
필터의 모든 요소를 합하면 0이 되도록 필터를 디자인하면, __0을 감지 레벨의 기준으로 세울 수 있어__ 편리하다.

필터를 적용하면 출력 이미지의 크기가 작아진다.
-> 해결책으로 __패딩(Padding)__ 이라는 방법을 사용한다.

* 패딩(Padding) : 필터를 적용하기 전에 0등의 고정된 요소로 주위를 부풀려두는 방법.
ex) 3x3필터에서는 폭 1의 패딩을 하면 출력 이미지 크기가 변하지 않는다.
ex) 5x5사이즈 필터를 사용하는 경우에는 패딩을 2로 하면 좋을 것.


* 스트라이드(Stride) : 필터를 적용할 때 이동하는 간격. 스트라이드를 크게 하면 출력 이미지의 사이즈가 작아진다.

<br>

<br>

## 8.5 합성곱 신경망

* 합성곱 신경망(CNN, Convolution Neural Network) : 필터를 사용한 신경망

CNN은 필터 자체를 학습시킨다.

필터 8장을 사용한 CNN을 만들어보자.
3x3 사이즈, 패딩 1, 스트라이드 1의 필터 8장 적용.
필터는 28x28x8의 3차원 배열이 된다.
-> 1차원으로 길이가 6272인 배열로 전개하고, 전결합으로 10개의 출력층 뉴런에 결합한다.

CNN을 케라스로 구현하자.

```python
import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import time


model = Sequential()
model.add(Conv2D(8, (3, 3), padding='same',
                 input_shape=(28, 28, 1), activation='relu')) # (A)
model.add(Flatten()) # (B)
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
startTime = time.time()
history = model.fit(x_train, y_train, batch_size=1000, epochs=20,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))
```

* (A) 합성곱층 `Conv2D()`를 __model__ 에 추가한다.
`model.add(Conv2D(8, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))`
  * (8, (3, 3)) : 3x3의 필터를 8개 사용한다는 의미
  * padding='same' : 출력 크기가 변하지 않도록 패딩을 추가해 처리한다는 뜻
  * input_shape=(28, 28 1) : 입력 이미지의 크기( 마지막1: 흑백이미지라서. 컬러라면 3)
  * activation='relu' : 필터를 적용한 후에 ReLU활성화 함수를 적용하겠음.
기본적으로 바이어스 입력도 지정되어있다. 바이어스는 각 필터에 1변수씩 할당된다. 또한, 필터의 학습 전의 초기값은 임의로 설정되며, 바이어스의 초기값은 0으로 설정된다.
* (B) 출력 변환
`model.add(Flatten())`
합성곱층의 출력 : (배치 수, 필터 수, 출력이미지세로폭, 출력이미지가로폭)
4차원인 합성곱층의 출력을 다음 출력층에 넣으려면 (배치 수, 필터수x출력이미지세로폭x출력이미지세로폭)의 _2차원으로 변환_ 해야한다.

실행결과를 보자!

```python
show_prediction()
plt.show()
```

@@@@@@@@@@@@이미지

잘못 인식한 이미지가 단 2개일 정도로 학습이 잘 진행되었다.

학습에서 획득한 8장의 필터를 살펴보자.

```python
plt.figure(1, figsize=(12, 2.5))
plt.gray()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.subplot(2, 9, 10)
id_img = 12
x_img = x_test[id_img, :, :, 0]
img_h = 28
img_w = 28
x_img = x_img.reshape(img_h, img_w)
plt.pcolor(-x_img)
plt.xlim(0, img_h)
plt.ylim(img_w, 0)
plt.xticks([], "")
plt.yticks([], "")
plt.title("Original")


w = model.layers[0].get_weights()[0] # (A)
max_w = np.max(w)
min_w = np.min(w)
for i in range(8):
    plt.subplot(2, 9, i + 2)
    w1 = w[:, :, 0, i]
    w1 = w1.reshape(3, 3)
    plt.pcolor(-w1, vmin=min_w, vmax=max_w)
    plt.xlim(0, 3)
    plt.ylim(3, 0)
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("%d" % i)
    plt.subplot(2, 9, i + 11)
    out_img = np.zeros_like(x_img)
    # 필터 처리
    for ih in range(img_h - 3):
        for iw in range(img_w - 3):
            img_part = x_img[ih:ih + 3, iw:iw + 3]
            out_img[ih + 1, iw + 1] = \
            np.dot(img_part.reshape(-1), w1.reshape(-1))
    plt.pcolor(-out_img)
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0)
    plt.xticks([], "")
    plt.yticks([], "")
plt.show()
```

@@@@@@@@@@@@@@@ 이미지

<br>

<br>

## 8.6 풀링(Pooling)

합성곱층에서 이미지의 위치가 약간만 어긋나도 출력이 달라지게된다.
-> __'풀링 처리'__ 로 문제를 해결할 수 있다.

'최대 풀링(max pooling)' : 입력 이미지 내의 2x2의 작은 영역에 착안하여 __가장 큰 값을 출력값으로__ 한다. 작은 공간은 스트라이드=2로 이동해서 동일한 처리를 반복.-> 출력 이미지의 사이즈가 가로, 세로 1/2로 줄어듬.

이렇게하여 얻은 __출력 이미지는 입력 이미지가 가로 세로로 어긋나도 거의 변하지 않는 성질을__ 갖는다.

최대 풀링 외에 '평균 풀링'이 있다. 평균 풀링은 작은 영역의 수치의 평균값을 출력값으로 한다.

<br>

<br>

## 8.7 드롭아웃

네트워크 학습을 개선하는 방법으로 __'드롭아웃(Dropout)'__ 이라는 방법이 제안되었다.

__학습 시에 입력층의 유닛과 중간층 뉴런을 일정 확률 p(p<1)로 선택하여 나머지를 무효화하는 방법.__
무효화된 뉴런은 존재하지 않는 것으로 하여 학습을 갱신한다. 미니 배치마다 뉴런을 뽑아 이 절차를 반복한다.

학습 후 예측하는 경우에는 모든 뉴런이 사용되므로 전체 참가가 되면 출력이 커져버린다. 이 문제를 해결하기 위해 드롭아웃을 한 층의 출력 대상의 가중치를 p배(p<1)해서 작게 줄여서 계산을 맞춘다.

드롭아웃은 __여러 네트워크를 각각 학습시켜 예측 시에 네트워크를 평균화해 합치는 효과가__ 있다.

<br>

<br>

## 8.8 MNIST 인식 네트워크 모델

"합성곱 네트워크 + 풀링 + 드롭아웃"으로 계층의 수를 늘리고, 모두를 갖추고 있는 네트워크를 구축해보자.

@@@@@@@@@ 이미지

1. 1층, 2층에서 합성곱층을 연속시킨다.
  1층째의 합성곱층은 16장의 필터를 사용->출력: 28x28의 이미지 16개. 이것을 28x28x16의 3차원 배열의 데이터로 간주한다.
  2층째의 합성곱은 이 3차원 배열 데이터에 대해 수행된다. 3x3의 1장의 필터는 실질적으로 3x3x16의 배열로 정의된다.출력은 28x28x16의 블록이다. 16의 깊이분에 별도의 필터가 할당되어 독립적으로 처리되는 인상을 갖는다.3x3x16의 크기를 가지는 필터가 32개 있는 두 번째 레이어의 합성곱층.

2. 3층째는 2x2 맥스 풀링(최대 풀링)층으로, 이미지의 가로 세로의 크기는 절반인 14x14가 되고,
3. 4층째에서 한 번 더 합성곱층이 온다. 4층에서의 필터의 개수는 64장이다. 출력 이미지는 14x14사이즈 64장이 된다. 매개 변수의 수는 3x3x32x64이다.
4. 5층에서 맥스 풀링에 의해 이미지 크기가 7x7이 된다. + 드롭아웃
5. 6층에서 128개의 전결합 + 드롭아웃
6. 7층에서 출력이 10개인 전결합층이 된다.

```python
import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import startTime

# ------------------- 네트워크 레이어 생성
model = Sequential()
# 1층 합성곱
model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu'))
# 2층 합성곱
model.add(Conv2D(32, (3, 3), activation='relu'))
# 3층 맥스 풀링
model.add(MaxPooling2D(pool_size=(2, 2)))
# 4층 합성곱
model.add(Conv2D(64, (3, 3), activation='relu'))
# 5층 맥스 풀링 + 드롭 아웃
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))      # 0.25의 확률로 뉴런 선택.
model.add(Flatten())          # 1차원으로 Flatten
# 6층 128개의 전결합 + 드롭 아웃
model.add(Dense(128, activaton='relu'))
model.add(Dropout(0.25))      # 0.25의 확률로 뉴런 선택.
# 7층 출력이 10개인 전결합층
model.add(Dense(num_classes, activation='softmax'))   # num_classes=10이다.



model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])


startTime = time.time()

history = model.fit(x_train, y_train, batch_size=1000, epochs=20, verbose=1, validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print("Computation time: {0:3f} sec.".format(time.time()-startTime))
====
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 57s 958us/step - loss: 0.6237 - acc: 0.8080 - val_loss: 0.1167 - val_acc: 0.9645
Epoch 2/20
60000/60000 [==============================] - 56s 931us/step - loss: 0.1296 - acc: 0.9614 - val_loss: 0.0622 - val_acc: 0.9799
Epoch 3/20
60000/60000 [==============================] - 56s 930us/step - loss: 0.0860 - acc: 0.9740 - val_loss: 0.0465 - val_acc: 0.9860
Epoch 4/20
60000/60000 [==============================] - 56s 926us/step - loss: 0.0676 - acc: 0.9790 - val_loss: 0.0377 - val_acc: 0.9884
Epoch 5/20
60000/60000 [==============================] - 56s 926us/step - loss: 0.0557 - acc: 0.9829 - val_loss: 0.0319 - val_acc: 0.9890
Epoch 6/20
60000/60000 [==============================] - 56s 928us/step - loss: 0.0499 - acc: 0.9843 - val_loss: 0.0290 - val_acc: 0.9907
Epoch 7/20
60000/60000 [==============================] - 56s 929us/step - loss: 0.0452 - acc: 0.9856 - val_loss: 0.0283 - val_acc: 0.9900
Epoch 8/20
60000/60000 [==============================] - 55s 919us/step - loss: 0.0388 - acc: 0.9878 - val_loss: 0.0245 - val_acc: 0.9916
Epoch 9/20
60000/60000 [==============================] - 55s 920us/step - loss: 0.0339 - acc: 0.9896 - val_loss: 0.0259 - val_acc: 0.9909
Epoch 10/20
60000/60000 [==============================] - 55s 919us/step - loss: 0.0340 - acc: 0.9894 - val_loss: 0.0240 - val_acc: 0.9915
Epoch 11/20
60000/60000 [==============================] - 55s 922us/step - loss: 0.0300 - acc: 0.9904 - val_loss: 0.0247 - val_acc: 0.9918
Epoch 12/20
60000/60000 [==============================] - 55s 918us/step - loss: 0.0275 - acc: 0.9911 - val_loss: 0.0249 - val_acc: 0.9919
Epoch 13/20
60000/60000 [==============================] - 55s 917us/step - loss: 0.0250 - acc: 0.9918 - val_loss: 0.0213 - val_acc: 0.9922
Epoch 14/20
60000/60000 [==============================] - 55s 920us/step - loss: 0.0237 - acc: 0.9923 - val_loss: 0.0218 - val_acc: 0.9926
Epoch 15/20
60000/60000 [==============================] - 55s 916us/step - loss: 0.0219 - acc: 0.9930 - val_loss: 0.0231 - val_acc: 0.9922
Epoch 16/20
60000/60000 [==============================] - 55s 917us/step - loss: 0.0212 - acc: 0.9934 - val_loss: 0.0258 - val_acc: 0.9912
Epoch 17/20
60000/60000 [==============================] - 55s 921us/step - loss: 0.0199 - acc: 0.9931 - val_loss: 0.0240 - val_acc: 0.9924
Epoch 18/20
60000/60000 [==============================] - 55s 922us/step - loss: 0.0196 - acc: 0.9938 - val_loss: 0.0251 - val_acc: 0.9910
Epoch 19/20
60000/60000 [==============================] - 55s 921us/step - loss: 0.0182 - acc: 0.9940 - val_loss: 0.0254 - val_acc: 0.9921
Epoch 20/20
60000/60000 [==============================] - 55s 918us/step - loss: 0.0161 - acc: 0.9946 - val_loss: 0.0225 - val_acc: 0.9929
Test loss: 0.022540838093019558
Test accuracy: 0.9929
Computation time:1113.930 sec
```

```python
show_prediction()
plt.show()
```

@@@@@@@@@@@@@@@ 이미지

MNIST 데이터보다 더 큰 크기의 자연의 이미지를 처리하거나 많은 카테고리를 다루는 경우에는 층의 심층화, 합성곱, 풀링, 드롭아웃의 효과가 더욱 강력하게 발휘될 것이다.


> [https://www.tensorflow.org/tutorials/images/cnn](https://www.tensorflow.org/tutorials/images/cnn) 를 보고 이해에 도움을 받자!
> Conv2D와 MaxPooling2D층의 출력은 (높이, 너비, 채널)크기의 3D텐서이다. 네트워크가 깊어질수록 높이와 너비 차원이 감소하는 경향을 가진다. Conv2D 층에서 출력 채널의 개수는 첫 번째 매개변수에 의해 결정된다.
> 모델을 완성하기 위해서 마지막 합성곱 층의 출력 텐서를 하나 이상의 Dense층에 주입하여 분류를 수행한다. Dense 층은 벡터(1D)를 입력으로 받으므로 Flatten()명령으로 3D텐서를 1D로 펼친다.
