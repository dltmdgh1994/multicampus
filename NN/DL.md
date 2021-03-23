# 딥러닝의 역사

<img src="md-images\artifical_neural_network.PNG" alt="artifical_neural_network" style="zoom:75%;" />

Y = f(W * X + b ) => f : activation function

입력값들의 선형 결합으로서 출력값을 추정하려고 하는 것이 인공 신경망의 원리



1.  하지만, XOR 문제를 신경망 하나로는 해결할 수 없었음. 

   XOR 문제를 해결 하기 위해 인공 신경망을 좌우, 상하로 깊게 쌓았음. (Multi layer)

    위에서 언급한 Activation Function로 비선형성인 Sigmoid 등의 함수를 사용.

2.  그러자 계산량이 굉장히 많아서 학습이 안됨. 

   BackPropagation으로 학습 했더니 학습에 필요한 연산량이 급격히 줄어들면서 학습이 가능해짐.

   * **BackPropagation(오차 역전파)**

     출력층부터 **역방향으로 에러를 전파**시키면서 최적의 학습 결과를 찾는 것

     출력부터 입력 쪽으로 순차적으로 cost function에 대한 편미분을 구하고, 얻은 편미분 값을 이용해서 w와 b의 값을 갱신시키고, 모든 훈련 데이터에 대해서 이 작업을 반복적으로 수행하면, 훈련 데이터에 최적화된 w와 b를 찾을 수 있다.

     즉, 역전파를 하기 위해서는 기존에 설정되어 있는 임의의 w와 b로 에러와 cost function을 구하고**(feed foward)**, 이를 통해 backpropagation을 진행한다.

3. Multi-layer가 3층 이상 쌓이면 Vanishing Gradient 문제가 생기면서 학습이 잘 안됨.

    Activation Function으로 ReLU를 사용했더니 위의 문제가 사라지면서 학습이 가능해짐



# DNN

![DNN](md-images\DNN.PNG)

Deep Neural Network로 은닉층을 2개 이상 지닌 학습 방법을 지칭함



# CNN

![CNN](md-images\CNN.PNG)

Convolutional Neural Network로 기존 DNN을 이용해 이미지를 학습하는데는 여러가지 한계점이 존재하기 때문에, 사람이 그림을 판별하듯 이미지의 특징을 학습하여 예측하는 방법



## Convolution(합성곱)

<img src="md-images\convolution.PNG" alt="convolution" style="zoom:75%;" />



기존 인공 신경망은 1차원 텐서인 벡터로 변환하여 입력층으로 사용해야 한다. 이러면 이미지의 **공간적인 구조 정보가 유실**되기 때문에 CNN에서는 이러한 이미지의 특징을 학습하기 위해 1차원으로 변환하지 않는다.



![convolution2](md-images\convolution2.jpg)

이러한 과정을 거쳐 하나의 **Feature Map**을 생성한다.

## Filter(Kernel)

합성곱 연산 과정에서 이미지의 특징을 찾아내기 위한 공용 parameter

일반적으로 3x3, 4x4의 정방형 형태로 정의

**결국 Filter안의 구성요소들이 CNN의 학습대상** => Filter를 통해 Feature Map을 추출

이미지의 RGB와 같이 channel이 여러 개면 여러 개의 Filter를 통해 여러 개의 Feature Map을 추출

![filter](md-images\filter.PNG)



## Stride

합성곱 연산 시 커널이 이미지에서 움직이는 이동 범위

5x5 이미지에 3x3 커널이 stride가 2일 경우 2x2의 Feature Map 추출

![stride](md-images\stride.PNG)

즉, stride 값이 커지면 Feature Map 크기가 작아진다.



## Padding

합성곱 연산 시 Feature Map은 기존 이미지(입력)에 비해 작아진다. 이러한 연산을 여러번 수행하면 Feature Map이 너무 작아지기 때문에 이걸 방지하기 위해 Padding을 사용

![padding](md-images\padding.PNG)

입력 데이터 외곽에 0으로 채우는 것을 **zero padding** => 특성에 영향을 주지 않아 많이 사용



## Pooling

합성곱 연산 중간중간에 학습하는 데이터의 양을 줄이고 특정 feature를 강조하기 위해 사용

![pooling](md-images\pooling.PNG)0

