import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import nn
from torch import optim

# MNIST 데이터셋 로드
mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)

## 첫 번째 샘플 확인
print(mnist.data[0])  # 첫 번째 샘플의 픽셀 값 (784차원 벡터)
print(mnist.target[0])  # 첫 번째 샘플의 레이블 (숫자 5)

## 레이블 데이터 타입을 정수형으로 변환
mnist.target = mnist.target.astype(np.int8)

# 훈련 데이터 준비
X = mnist.data / 255  # 이미지 데이터를 [0, 1] 구간으로 정규화 (70000, 784)
y = mnist.target  # 이미지의 실제 숫자 레이블 (0~9)

## 정규화된 데이터 출력
print(X[0])  # 첫 번째 샘플의 정규화된 픽셀 값

## 첫 번째 데이터 시각화
plt.imshow(X[0].reshape(28, 28), cmap="gray")
print("이 이미지 데이터의 레이블은 {:.0f}이다".format(y[0]))
# plt.show()

## 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 7, random_state=0
)

## 텐서로 변환
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

## TensorDataset 객체 생성
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

## DataLoader 객체 생성
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

# 모델 준비
## 784개의 입력 뉴런, 100개의 은닉 뉴런, 10개의 출력 뉴런을 가지는 다층 퍼셉트론 모델 정의
## ReLU 활성화 함수를 통해 비선형성을 추가
## 최종 출력은 10개의 클래스에 대한 점수를 제공
model = nn.Sequential()
model.add_module("fc1", nn.Linear(28 * 28 * 1, 100))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2", nn.Linear(100, 100))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(100, 10))

print(model)

## 손실 함수와 옵티마이저 정의
criterion = (
    nn.CrossEntropyLoss()
)  # 다중 클래스 분류 문제에서 사용하는 크로스 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 옵티마이저

# 모델 학습
epochs = 3

for epoch in range(epochs):
    for data, targets in loader_train:
        y_pred = model(data)  # 순전파 연산으로 예측값 계산
        loss = criterion(y_pred, targets)  # 손실 함수로 비용 계산

        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
        loss.backward()  # 역전파 연산으로 기울기 계산
        optimizer.step()  # 옵티마이저를 통해 파라미터 업데이트

    print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch + 1, 3, loss.item()))

# 테스트 데이터에 대한 평가
model.eval()  # 모델을 평가 모드로 전환
correct = 0

## 테스트 데이터에 대한 예측 수행
with torch.no_grad():  # 기울기 계산 비활성화
    for data, targets in loader_test:

        outputs = model(data)  # 데이터를 입력하고 출력을 계산

        # 추론 계산
        _, predicted = torch.max(outputs.data, 1)  # 확률이 가장 높은 레이블이 무엇인지 계산
        correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 정답과 일치한 경우 정답 카운트를 증가

## 정확도 출력
data_num = len(loader_test.dataset)  # 데이터 총 건수
print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))

## 임의의 테스트 샘플에 대한 예측 결과 시각화
index = 2018

model.eval()  # 모델을 평가 모드로 전환
data = X_test[index]
output = model(data)  # 모델에 입력 데이터를 넣어 예측값 계산
_, predicted = torch.max(output.data, 0)  # 예측값에서 가장 높은 점수를 가진 레이블 계산

print("예측값: ", predicted.item())  # 예측된 레이블 출력

X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다".format(y_test[index]))
plt.show()