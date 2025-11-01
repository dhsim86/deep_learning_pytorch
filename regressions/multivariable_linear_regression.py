import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 선언
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설 및 비용 함수, 옵티마이저 선언 후 경사하강법을 2,000회 반복
## optimizer는 SGD로 설정, 학습률은 1e-5
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # 가설 설정
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # 비용 함수 설정, cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 비용 함수로 H(x) 개선
    optimizer.zero_grad()   # gradient 초기화
    cost.backward()         # 비용 함수를 미분하여 gradient 계산
    optimizer.step()        # w와 b를 gradient 방향으로 업데이트

    # 50번마다 로그 출력
    if epoch % 50 == 0:
        print('Epoch {:4d}/{} w1: {:.3f}, w2: {:.3f}, w3: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))