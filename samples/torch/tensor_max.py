import torch

# 1. 원소의 최대값 구하기
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"2차원 텐서의 최대값: {t.max()}")  # 4.0

print("-" * 30)

# 2. 차원 제거 후 최대값 구하기
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"차원 제거 후 최대값 및 shape: {t.max(dim=0)}")  # [3, 4] (첫 번째 차원을 제거 후 최대값, 각 열의 최대값)
print(f"차원 제거 후 최대값 및 shape: {t.max(dim=1)}")  # [2, 4] (두 번째 차원을 제거 후 최대값, 각 행의 최대값)
