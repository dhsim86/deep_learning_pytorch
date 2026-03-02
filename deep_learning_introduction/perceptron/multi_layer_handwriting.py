import matplotlib.pyplot as plt  # 시각화를 위한 맷플롯립

# 사이킷런에서 제공하는 분류용 손글씨 숫자 데이터셋 로드
## 0~9까지의 숫자를 손으로 쓴 이미지 데이터
## 각 이미지는 0~15까지의 명암을 가지는 8x8 픽셀로 구성되어 있음
from sklearn.datasets import load_digits

digits = load_digits()  # 1,979개의 이미지 데이터 로드

print(digits.images[0])  # 첫 번째 이미지 데이터 출력 (8x8 픽셀)
print(digits.target[0])  # 첫 번째 이미지의 실제 숫자 레이블 출력 (0~9)

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):  # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("sample: %i" % label)
plt.tight_layout()
plt.show()

for i in range(5):
    print(i, "번 인덱스 샘플의 레이블 : ", digits.target[i])

# 훈련 데이터와 레이블 준비
X = digits.data  # data에는 이미지 데이터를 64차원 벡터로 변환한 상태
Y = digits.target  # 이미지의 실제 숫자 레이블 (0~9)

import torch
import torch.nn as nn
from torch import optim

# 모델 정의: 순차적인 레이어 구조

## 다층 퍼셉트론 모델 정의
model = nn.Sequential(
    nn.Linear(
        64, 32
    ),  # 입력층 -> 첫번째 은닉층, 입력 데이터의 특성 64개를 받아 32개의 출력을 생성
    nn.ReLU(),  # 활성화 함수: ReLU, 비선형성을 추가
    nn.Linear(32, 16),  # 두번째 은닉층, 32개의 입력을 받아 16개의 출력을 생성
    nn.ReLU(),  # 활성화 함수: ReLU
    nn.Linear(
        16, 10
    ),  # 두번째 은닉층 -> 출력층, 16개의 입력을 받아 10개의 클래스로 출력
)

# 입력 데이터 X와 레이블 Y를 텐서로 변환
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

# 손실 함수와 옵티마이저 정의
criterion = (
    nn.CrossEntropyLoss()
)  # 다중 클래스 분류 문제에서 사용하는 크로스 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters())  # Adam 옵티마이저

losses = []  # 손실 값을 저장할 리스트

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    # 순전파
    outputs = model(X)  # 모델에 입력 X를 넣어 예측값 계산

    # 손실 함수 계산
    loss = criterion(outputs, Y)  # 예측값과 실제 레이블 Y를 비교하여 손실 계산

    # 역전파 및 최적화
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()  # 손실 함수에 대한 기울기 계산
    optimizer.step()  # 가중치 갱신

    losses.append(loss.item())  # 손실 값을 리스트에 추가하여 추적

    if (epoch + 1) % 10 == 0:  # 10 에폭마다 손실 출력
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 손실 값 시각화
plt.plot(losses)
plt.show()  # 손실 값의 변화를 시각적으로 확인