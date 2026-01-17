import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터 준비
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

print("x_train shape:", x_train.shape)  # (8, 4)
print("y_train shape:", y_train.shape)  # (8, 1)

# 원핫 인코딩 준비
y_one_hot = torch.zeros(8, 3) # 레이블이 3개이므로
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print("원핫 인코딩 결과:\n", y_one_hot)

# 가중치 준비
## 입력 크기 (8, 4), 출력 크기 (8, 3) -> (4, 3) 크기의 가중치 필요
W = torch.zeros((4, 3), requires_grad=True)  # (특징 수, 클래스 수)
b = torch.zeros((1, 3), requires_grad=True)  # (클래스 수)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

# 학습
np_epochs = 10000
for epoch in range(np_epochs + 1):

    # 가설(H(x)) 계산
    z = x_train.matmul(W) + b  # (8, 4) @ (4, 3) = (8, 3)
    hypothesis = F.softmax(z, dim=1)

    # 비용함수 계산
    loss = -(y_one_hot * torch.log(hypothesis)).sum(dim=1).mean()

    # 비용함수로 H(x) 개선
    optimizer.zero_grad()  # gradient 초기화
    loss.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 gradient 방향으로 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, np_epochs, loss.item()))