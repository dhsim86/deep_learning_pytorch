import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand(3, 5, requires_grad=True)
y = torch.randint(5, (3,)).long() # 0~4 사이의 정수 레이블 3개 생성

print("예측값(z):\n", z)
print("임의의 레이블:", y.tolist())

# nn.CrossEntropyLoss 테스트
## nn.CrossEntropyLoss는 내부적으로 소프트맥스 함수까지 포함함
criterion = nn.CrossEntropyLoss()
loss = criterion(z, y)
print("nn.CrossEntropyLoss 결과:", loss.item())
