import torch

# 1. 벡터 평균 구하기
t = torch.FloatTensor([1, 2, 3, 4, 5])
print(f"1차원 텐서의 평균: {t.mean()}")  # 3.0

print("-" * 30)

# 2. 행렬 평균 구하기
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"2차원 텐서의 평균: {t.mean()}")  # 2.5 (4개 원소의 평균)

print("-" * 30)

# 3. 차원 제거 후 평균 구하기
t = torch.FloatTensor([[1, 2], [3, 4]])
print(f"차원 제거 후 평균 및 shape: {t.mean(dim=0)} / {t.mean(dim=0).shape}")  # [2, 3] (첫 번째 차원을 제거 후 평균, 각 열의 평균)
print(f"차원 제거 후 평균 및 shape: {t.mean(dim=1)} / {t.mean(dim=1).shape}")  # [1.5, 3.5] (두 번째 차원을 제거 후 평균, 각 행의 평균)

