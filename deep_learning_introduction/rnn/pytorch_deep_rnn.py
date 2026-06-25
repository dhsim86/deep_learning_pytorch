import torch
import torch.nn as nn

# 깊은 순환 신경망을 파이토치로 구현시 nn.RNN() 인자인 num_layers에 은닉층 갯수를 지정

# 입력 텐서 정의
inputs = torch.Tensor(1, 10, 5)

# RNN 셀 생성
## num_layers를 2로 지정하여 층이 2개인 깊은 순환 신경망 정의
cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first=True)

## 출력 확인
outputs, _status = cell(inputs)

### 10번의 시점동안 8차원의 은닉 상태가 출력, 층이 1개일 때와 같음
print(outputs.shape) # torch.Size([1, 10, 8])

### 최종 time-step의 hidden_state (층의 갯수, 배치 크기, 은닉 상태 크기)
print(_status.shape) # torch.Size([2, 1, 8])