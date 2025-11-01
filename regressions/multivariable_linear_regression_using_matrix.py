import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("=============================================")

# 가중치와 편향 선언
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print("초기 가중치 w:", w)
print("가중치 w shape:", w.shape)
print("초기 편향 b:", b)
print("편향 b shape:", b.shape) 

print("=============================================")

# 가설 및 비용 함수, 옵티마이저 선언 후 경사하강법을 2,000회 반복
## optimizer는 SGD로 설정, 학습률은 1e-5
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # 가설 설정
    ## 편향 b는 브로드캐스팅 되어 샘플마다 더해짐
    hypothesis = x_train.matmul(w) + b  # 행렬 곱셈(matmul) 사용

    # 비용 함수 설정, cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 비용 함수로 H(x) 개선
    optimizer.zero_grad()   # gradient 초기화
    cost.backward()         # 비용 함수를 미분하여 gradient 계산
    optimizer.step()        # w와 b를 gradient 방향으로 업데이트

    # 50번마다 로그 출력
    if epoch % 50 == 0:
        print('Epoch {:4d}/{} hypothesis: {}, w: {}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), w.view(-1).tolist(), b.item(), cost.item()
        ))

print("=============================================")

# 임의의 새로운 데이터에 대한 예측
## with torch.no_grad(): 이 블록 안에서 실행되는 모든 연산에 대해 역전파(기울기 계산)을 비활성화
##  - 인퍼런스할 때는 가중치를 업데이트할 필요없으므로 메모리와 계산 절약을 위해 비활성화
with torch.no_grad():
    new_input = torch.FloatTensor([[75, 85, 72]]) # 새로운 입력 데이터 정의, 학습 데이터와 동일한 차원

    # 예측 수행, 학습 완료된 가중치를 갖고 연산
    prediction = new_input.matmul(w) + b # w, b는 학습 과정에서 얻어진 최적의 값
    print('Predicted value for input {}: {}'.format(new_input.squeeze().tolist(), prediction.item()))