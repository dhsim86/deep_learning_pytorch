import torch
import torch.nn as nn

input_size = 5 # 입력의 크기, 매 시점마다 들어가는 입력 크기
hidden_size = 8 # 은닉 상태의 크기, RNN의 하이퍼파라미터

# 입력 텐서 정의 (배치 크기 x 시점의 수 x 매 시점마다 들어가는 입력 크기)
inputs = torch.Tensor(1, 10, 5)

# RNN 셀 생성
## 시점마다의 입력 크기 및 은닉 상태 크기 지정
## batch_first를 통해 입력 텐선의 첫번째 차원이 배치 크기임을 알림
cell = nn.RNN(input_size, hidden_size, batch_first=True)

## 출력 확인
### 첫 번째 리턴값: 모든 시점(timesteps)의 은닉 상태
### 두 번째 리턴값: 마지막 시점의 은닉 상태
outputs, _status = cell(inputs)

### 10번의 시점동안 8차원의 은닉 상태가 출력됨
print(outputs.shape) # torch.Size([1, 10, 8])

### 최종 time-step의 hidden_state (층의 갯수, 배치 크기, 은닉 상태 크기)
print(_status.shape) # torch.Size([1, 1, 8])
