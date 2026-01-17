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

print(x_train.shape)  # (6, 2)
print(y_train.shape)  # (6, 1)

# 실제값 출력
print("실제값: ", y_train.squeeze().tolist())

# 가중치
W = torch.zeros((2, 1), requires_grad=True)  # (2, 1) 크기를 가져야 함
b = torch.zeros(1, requires_grad=True)

# 로지스틱 회귀의 가설
## e^x를 계산하는 함수는 torch.exp() 사용
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
# hypothesis = torch.sigmoid(x_train.matmul(W) + b)  # 동일한 결과, 파이토치에서 제공하는 시그모이드 함수 사용

# 학습전 예측값 출력
# 가중치와 편향이 모두 0이므로 예측값은 모두 0.5에 가까움
print("학습전 예측값: ", hypothesis.squeeze().detach().tolist())

# 비용함수
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
# losses = F.binary_cross_entropy(hypothesis, y_train)  # 동일한 결과, 파이토치에서 제공하는 이진 크로스 엔트로피 함수 사용
print("학습전 오차: ", losses.mean().item())

#######################################
# 학습 테스트

# 옵티마이저 설정
optimizer = optim.SGD([W, b], lr=1.0)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

    # 비용함수 계산
    loss = -(
        y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)
    ).mean()

    # 비용함수로 H(x) 개선
    optimizer.zero_grad()  # gradient 초기화
    loss.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 gradient 방향으로 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Loss: {:.6f}".format(epoch, nb_epochs, loss.item()))

# 학습 후 예측값 출력
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
print("학습후 예측값: ", hypothesis.squeeze().detach().tolist())
