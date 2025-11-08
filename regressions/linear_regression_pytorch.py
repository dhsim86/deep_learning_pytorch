import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 선언 및 초기화
## nn.Linear는 입력의 차원 및 출력의 차원을 인수로 받음
## 단순 선형 회귀이므로 input_dim과 output_dim이 모두 1
model = nn.Linear(1, 1)  # 입력 크기 1, 출력 크기 1

## 모델 파라미터 출력
print("초기 파라미터: ", list(model.parameters()))

# 옵티마이저 정의
## 경사 하강법 SGD 사용, 학습률은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

## 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # H(x) 계산, forward 연산
    prediction = model(x_train)

    # cost 계산
    ## 파이토치에서 제공하는 평균 제곱 오차 함수 사용
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()   # gradient 초기화
    cost.backward()         # 비용 함수를 미분하여 gradient 계산
    optimizer.step()        # W와 b를 gradient 방향으로 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} prediction: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, prediction.squeeze().tolist(), cost.item()
        ))

print("=============================================")
print("학습 후 파라미터: ", list(model.parameters()))

with torch.no_grad():
    # 임의의 새로운 데이터에 대한 예측
    new_input = torch.FloatTensor([[4], [5]]) # 새로운 입력 데이터 정의
    new_prediction = model(new_input)
    print("새로운 입력에 대한 예측:", new_prediction.squeeze().tolist())