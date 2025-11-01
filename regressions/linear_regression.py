import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 시드값 지정
torch.manual_seed(1)

##############################################
# 데이터셋 변수 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

print("훈련 데이터셋:")
print("입력:", x_train)
print("입력 크기:", x_train.shape) # (3, 1)
print("출력:", y_train)
print("출력 크기:", y_train.shape) # (3, 1)

# AutoGrad에 의해 자동으로 수행
## Optimizer는 단지 어떤 학습 파라미터를 업데이트할지 알고 있음
## cost가 그 변수들의 기울기를 계산하는 그래프의 끝점 역할
## AutoGrad 시스템이 이 둘을 간접적으로 이어준다.

##############################################
# 가중치와 편향 초기화
## 가중치 W를 0으로 초기화하고, 학습을 통해 값이 변경되는 변수로 명시

## requires_grad=True: 학습을 통해 계속 값이 변경되는 변수임을 의미
## - PyTorch의 연산 그래프안에서 "학습 가능한 변수"로 등록됨
## - W, b를 통해 어떤 계산을 하든 PyTorch는 연산을 모두 기록
## - 선형 회귀나 신경망과 같은 곳에서 사용되는 모든 파라미터는 이 기능이 적용됨
## - 이 것이 적용된 텐서에 연산을 하면 연산 그래프가 생성되고, backward 호출시 이 그래프로부터 자동으로 미분이 계산됨
W = torch.zeros(1, requires_grad=True) 
print("\n초기 가중치 W:", W)
print("가중치 W 크기:", W.shape) # (1,)

## 편향 b를 0으로 초기화하고, 학습을 통해 값이 변경되는 변수로 명시
b = torch.zeros(1, requires_grad=True)
print("초기 편향 b:", b)
print("편향 b 크기:", b.shape) # (1,)

##############################################
# 가설 세우기
hypothesis = x_train * W + b
print("\n가설:", hypothesis) # 가중치 W와 편향 b가 0이므로 모두 0으로 출력

##############################################
# 비용 함수 선언
## cost는 W, b를 포함한 연산(hypothesis)의 결과이므로, PyTorch는 내부적으로 cost는 W와 b에 의존하는 것으로 인식
## 이 시점에서 cost 객체의 내부에는 다음과 같은 연산 그래프가 포함됨
## - W -> hypothesis -> cost
## - b -> hypothesis -> cost
cost = torch.mean((hypothesis - y_train) ** 2) # 평균 제곱 오차(MSE), 56.0 / 3 = 18.6667
print("비용 함수:", cost)

##############################################
# 경사 하강법으로 최적화

## SGD: 경사 하강법의 일종, 확률적 경사 하강법(Stochastic Gradient Descent)
## lr: learning rate, 학습률
## 이 시점에서 optimizer는 내부적으로 [W, b]라는 파라미터 리스트를 갖는다.
optimizer = optim.SGD([W, b], lr=0.01) # 경사 하강법(SGD)으로 W와 b를 학습, 학습률은 0.01

# 기울기 초기화: 기울기를 초기화해야 새로운 가중치, 편향에 대해 새로운 기울기를 구할 수 있음
optimizer.zero_grad() # 기존에 계산된 기울기를 0으로 초기화

# 비용 함수를 미분하여 gradient 계산
## PyTorch는 연산 그래프를 거꾸로 따라가며, 각 학습 파라미터에 대한 미분값(∂cost/∂W, ∂cost/∂b)을 계산
## 자동으로 계산 후 W.grad, b.grad에 저장
cost.backward() # 가중치 W와 편향 b에 대해 기울기 계산

# 가중치와 편향 업데이트
## 호출되면 내부적으로 W = W - lr * W.grad, b = b - lr * b.grad 계산 수행
## cost.backward() 호출되었을 때, W.grad와 b.grad에 계산된 기울기가 저장되어 있으므로 
## 이를 바탕으로 W와 b 업데이트
optimizer.step() # 계산된 기울기를 바탕으로 W와 b 업데이트

# 업데이트 후
print("\n업데이트 후 가중치 W:", W)
print("업데이트 후 편향 b:", b)