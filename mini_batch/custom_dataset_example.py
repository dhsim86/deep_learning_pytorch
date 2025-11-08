import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 데이터셋 전처리
        self.x_data = [
            [73, 80, 75],
            [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70],
        ]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        # 데이터셋의 길이 (샘플 갯수) 반환
        return len(self.x_data)

    def __getitem__(self, idx):
        # 데이터셋에서 특정 인덱스의 샘플 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)  # 입력 크기 3, 출력 크기 1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)  # forward 연산

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()  # gradient 초기화
        cost.backward()  # 비용 함수를 미분하여 gradient 계산
        optimizer.step()  # W와 b를 gradient 방향으로 업데이트

        print(
            "Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
            )
        )

print("=============================================")

with torch.no_grad():
    # 임의의 새로운 데이터에 대한 예측
    new_input = torch.FloatTensor([[73, 80, 75]])  # 새로운 입력 데이터 정의
    new_prediction = model(new_input)
    print("새로운 입력에 대한 예측:", new_prediction.squeeze().tolist())