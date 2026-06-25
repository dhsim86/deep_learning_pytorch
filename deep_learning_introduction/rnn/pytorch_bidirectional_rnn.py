import torch
import torch.nn as nn

# 입력 텐서 정의 (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

# 양방햔 순환 신경망 구현시 bidirectional 인자에 True로 지정
cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first=True, bidirectional = True)

## 출력 확인
### 첫 번째 리턴값: 모든 시점(timesteps)의 은닉 상태
### 두 번째 리턴값: 마지막 시점의 은닉 상태
outputs, _status = cell(inputs)

# (배치 크기, 시퀀스 길이, 은닉 상태의 크기 x 2)
## 은닉 상태의 크기 값이 2배가 됨 (양방향 은닉 상태의 값들이 연결됨)
print(outputs.shape) # torch.Size([1, 10, 16])

# (층의 개수 x 2, 배치 크기, 은닉 상태의 크기)
## 양방향의 출력값을 층의 갯수만큼 쌓아 올림 (2 x 2)
## 정방햔 기준으로 마지막 시점의 은닉 상태
## 역방향 기준으로 첫 번째 시점의 은닉 상태
print(_status.shape) # torch.Size([4, 1, 8])