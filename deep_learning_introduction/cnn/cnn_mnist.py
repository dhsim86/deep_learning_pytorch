import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
    
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0  # 에포크당 평균 비용을 저장하기 위한 변수 초기화

    for X, Y in data_loader:  # 미니 배치 단위로 데이터를 꺼내옴. X는 입력 데이터, Y는 레이블
        # 이미지 데이터는 이미 (28x28) 크기를 가지므로, 별도의 reshape 필요 없음
        # 레이블 Y는 원-핫 인코딩이 아닌 정수형 클래스 레이블임
        X = X.to(device)  # 입력 데이터를 연산이 수행될 장치로 이동 (예: GPU)
        Y = Y.to(device)  # 레이블을 연산이 수행될 장치로 이동 (예: GPU)

        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
        hypothesis = model(X)  # 모델을 통해 예측값(hypothesis)을 계산 (순전파 연산)
        cost = criterion(hypothesis, Y)  # 예측값과 실제값 Y 간의 손실(cost) 계산
        cost.backward()  # 역전파 연산을 통해 기울기 계산
        optimizer.step()  # 옵티마이저를 통해 파라미터 업데이트

        avg_cost += cost / total_batch  # 현재 배치의 비용을 전체 배치 수로 나누어 누적

    # 에포크가 끝날 때마다 평균 비용 출력
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습을 진행하지 않을 것이므로 torch.no_grad() 사용
with torch.no_grad():
    # 테스트 데이터를 모델에 입력하기 위한 준비
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)  # 테스트 데이터셋의 크기를 맞추고, 연산을 위한 장치로 이동
    Y_test = mnist_test.test_labels.to(device)  # 테스트 데이터셋의 레이블을 연산을 위한 장치로 이동

    # 모델 예측 수행
    prediction = model(X_test)  # 테스트 데이터에 대해 모델이 예측한 결과값

    # 예측 결과와 실제 레이블 비교
    correct_prediction = torch.argmax(prediction, 1) == Y_test  # 예측된 클래스와 실제 레이블이 일치하는지 확인

    # 정확도 계산
    accuracy = correct_prediction.float().mean()  # 정확도를 계산하기 위해 일치하는 예측의 평균을 구함
    print('Accuracy:', accuracy.item())  # 정확도를 출력