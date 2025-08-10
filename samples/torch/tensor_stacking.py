import torch

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

# 1. stack 연산
print(f"stack 결과:\n{torch.stack([x, y, z])}")
print(f"stack은 unsqueeze와 cat으로 동일하게 구현 가능:\n{torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0 )}")

# 2. 열방향으로 stack
print(f"열방향으로 stack:\n{torch.stack([x, y, z], dim=1)}")