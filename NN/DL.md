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



# CNN

![CNN](md-images\CNN.PNG)