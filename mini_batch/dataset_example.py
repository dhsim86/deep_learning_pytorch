import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

# TensorDataset을 사용하여 데이터셋 정의
x_train = torch.FloatTensor(
    [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)

# DataLoader를 사용하여 미니 배치로 데이터 로드
## batch_size는 통상적으로 2의 배수로 설정
## shuffle=True로 설정하면 매 epoch마다 데이터셋을 섞어 학습되는 순서를 바꿈
##   -> 모델이 데이터셋 순서에 익숙해지는 것을 방지하도록 학습
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 및 옵티마이저 정의
model = nn.Linear(3, 1)  # 입력 크기 3, 출력 크기 1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        print("batch_idx:", batch_idx)
        print("samples:", samples)

        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)  # forward 연산

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()  # gradient 초기화
        cost.backward()  # 비용 함수를 미분하여 gradient 계산
        optimizer.step()  # W와 b를 gradient 방향으로 업데이트

        print("Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()))

print("=============================================")

with torch.no_grad():
    # 임의의 새로운 데이터에 대한 예측
    new_input = torch.FloatTensor([[73, 80, 75]])  # 새로운 입력 데이터 정의
    new_prediction = model(new_input)
    print("새로운 입력에 대한 예측:", new_prediction.squeeze().tolist())