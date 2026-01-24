import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 150
batch_size = 100

# MNIST dataset
## MNIST_data 폴더에 데이터가 저장됨
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,   # True이면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 데이터를 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,   # False이면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
## 마지막 배치를 버리는 이유
## -> 데이터 샘플이 1,000개일 때 배치크기가 128이면 마지막 배치는 104개가 남는다.
## -> 만약 마지막 배치를 쓰면 다른 미니 배치보다 갯수가 적으므로, 상대적으로 과대 평가된다.
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size, # 배치 크기는 100
                         shuffle=True,
                         drop_last=True) # 마지막 배치를 버릴지 여부

# 모델 정의
## MNIST data image of shape 28 * 28 = 784
## 10 -> 0 ~ 9 digits
## bias는 True로 설정하여 편향을 사용
## to(device) -> 모델의 매개변수를 지정한 장치의 메모리로 보낸다. GPU 사용시 지정해줘야 함
linear = nn.Linear(784, 10, bias=True).to(device)

# 비용함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 소프트맥스 함수와 크로스 엔트로피 비용함수를 합쳐놓은 함수
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1) # 경사하강법

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader) # 배치의 개수

    for X, Y in data_loader:
        # X, Y는 GPU에 배치
        # 배치 크기가 100이므로 X는 (100, 784)의 텐서가 됨
        X = X.view(-1, 28 * 28).to(device) # (100, 1, 28, 28) -> (100, 784)
        # 레이블은 0 ~ 9 사이의 정수
        Y = Y.to(device)

        optimizer.zero_grad() # gradient 초기화

        hypothesis = linear(X) # H(x) 계산
        cost = criterion(hypothesis, Y) # 비용함수 계산

        cost.backward() # 비용함수를 미분하여 gradient 계산
        optimizer.step() # W와 b를 gradient 방향으로 업데이트

        avg_cost += cost / total_batch # 배치의 비용 누적

    print('Epoch: {:04d}, Cost: {:.9f}'.format(epoch + 1, avg_cost.item()))

print('Learning Finished!')

# 테스트 데이터를 사용해 모델을 테스트
with torch.no_grad(): # 테스트시에는 gradient 계산을 하지 않음

    # 테스트 데이터 X, Y를 GPU에 배치
    # view(-1, 28 * 28) -> (10000, 784), 각 행마다 일렬로 펴줌
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device) # (10000, 784)
    Y_test = mnist_test.test_labels.to(device) # (10000)

    prediction = linear(X_test) # 모델의 예측값
    # 각 이미지에 대해 예측된 클래스 레이블(0 ~ 9일 확률) 반환해서 실제값과 비교
    correct_prediction = torch.argmax(prediction, 1) == Y_test # 예측값과 실제값 비교
    accuracy = correct_prediction.float().mean() # 전체 데이터셋에 대한 정확도 계산
    print('Accuracy:', accuracy.item())

    # MINIST 테스트 데이터에서 무작위로 하나를 뽑아 예측
    r = random.randint(0, len(mnist_test) - 1) # 0 ~ 9999 사이의 정수 중 무작위로 하나 선택
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device) # (1, 784)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device) # (1)
    print("Label: ", Y_single_data.item()) # 실제 레이블 출력
    single_prediction = linear(X_single_data) # 모델의 예측값
    print("Prediction: ", torch.argmax(single_prediction, 1).item()) # 예측 레이블 출력

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()