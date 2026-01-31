import torch
import torch.nn as nn

# GPU 연산 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# XOR 문제 데이터셋 정의
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) # (4, 2)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device) # (4, 1)

# 모델 정의 (이진 분류)
linear = nn.Linear(2, 1, bias=True) # 입력 2, 출력 1
sigmoid = nn.Sigmoid() # 활성화 함수로 시그모이드 사용
model = nn.Sequential(linear, sigmoid).to(device) # linear -> sigmoid

# 비용 함수와 옵티마이저 정의
## BCELoss: 이진 분류 문제에서 사용하는 크로스 엔트로피 손실 함수
criterion = nn.BCELoss().to(device) # 이진 크로스 엔트로피 비용 함수
optimizer = torch.optim.SGD(model.parameters(), lr=1) # 경사 하강법

# 모델 학습
for epoch in range(10001):
    # 순전파
    hypothesis = model(X) # 모델에 입력 X를 넣어 예측값 계산

    # 비용 함수 계산
    cost = criterion(hypothesis, Y)

    # 역전파
    optimizer.zero_grad() # 기울기 초기화
    cost.backward() # 비용 함수에 대한 기울기 계산
    optimizer.step() # 가중치 갱신

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/10000, Cost: {cost.item():.4f}')

with torch.no_grad(): # 기울기 계산 비활성화
    # 모델 평가
    hypothesis = model(X) # 모델에 입력 X를 넣어 예측값 계산 (시그모이드 출력이므로 0~1 사이 값)
    predicted = (hypothesis > 0.5).float() # 0.5를 기준으로 0과 1로 이진 분류
    accuracy = (predicted == Y).float().mean() # 정확도 계산

    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())