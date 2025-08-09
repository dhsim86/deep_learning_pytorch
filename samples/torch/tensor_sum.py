import torch

# 1. 단순 덧셈
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"2차원 텐서의 합: {t.sum()}")  # 10.0

print("-" * 30)

# 2. 차원 제거 후 덧셈
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"차원 제거 후 덧셈 및 shape: {t.sum(dim=0)} / {t.sum(dim=0).shape}")  # [4, 6] (첫 번째 차원을 제거 후 합, 각 열의 합)
print(f"차원 제거 후 덧셈 및 shape: {t.sum(dim=1)} / {t.sum(dim=1).shape}")  # [3, 7] (두 번째 차원을 제거 후 합, 각 행의 합)
