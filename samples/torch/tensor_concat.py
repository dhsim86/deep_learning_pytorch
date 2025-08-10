import torch

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(f"0번째 차원으로 concatenate:\n{torch.cat([x, y], dim=0)}")
print(f"1번째 차원으로 concatenate:\n{torch.cat([x, y], dim=1)}")