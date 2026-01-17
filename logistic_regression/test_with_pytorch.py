import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터 준비
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.Sequential을 통해 nn.Module을 차례로 쌓아 모델 구성
## 로지스틱 회귀에서는 선형 변환 + 시그모이드 함수 사용
## nn.Sequential은 여러 층을 쌓는 인공신경망을 구현할 때 사용
model = nn.Sequential(
    nn.Linear(2, 1), nn.Sigmoid()  # 입력 크기 2, 출력 크기 1  # 시그모이드 함수
)

# 실제값 출력
print("실제값: ", y_train.squeeze().tolist())

# 예측값 출력
print("학습전 예측값: ", model(x_train).squeeze().detach().tolist())

#############
# 훈련

# 옵티마이저 설정
optimizer = optim.SGD(model.parameters(), lr=1.0)

np_epochs = 1000
for epoch in range(np_epochs + 1):
    # H(x) 계산
    hypothesis = model(x_train)

    # 비용함수 계산
    loss = F.binary_cross_entropy(hypothesis, y_train)

    # 비용함수로 H(x) 개선
    optimizer.zero_grad()  # gradient 초기화
    loss.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 gradient 방향으로 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print(
            "Epoch {:4d}/{} Loss: {:.6f}".format(epoch, np_epochs, loss.item())
        )

# 학습 후 예측값 출력
print("학습후 예측값: ", model(x_train).squeeze().detach().tolist())