import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)  # 입력 크기 3, 출력 크기 1

## 모델 파라미터 출력
print("초기 파라미터: ", list(model.parameters()))

# 옵티마이저 정의
## 경사 하강법 SGD 사용, 학습률은 1e-5
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 학습률을 0.01로 설정하면 발산함

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
            epoch, nb_epochs, prediction.squeeze().detach().tolist(), cost.item()
        ))

print("=============================================")
print("학습 후 파라미터: ", list(model.parameters()))

with torch.no_grad():
    # 임의의 새로운 데이터에 대한 예측
    new_input = torch.FloatTensor([[73, 80, 75]]) # 새로운 입력 데이터 정의
    new_prediction = model(new_input)
    print("새로운 입력에 대한 예측:", new_prediction.squeeze().tolist())