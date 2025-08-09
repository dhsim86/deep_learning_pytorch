import torch

# 1. 행렬 곱셈 (2, 2) * (2, 1) => (2, 1)
print("1. 행렬 곱셈 (2, 2) * (2, 1) => (2, 1)")

m1 = torch.FloatTensor([[1, 2], [3, 4]])  # (2, 2)
m2 = torch.FloatTensor([[1], [2]])  # (2, 1)
print(f"shape of m1: {m1.shape}, shape of m2: {m2.shape}")
print(f"m1.matmul(m2) 결과\n: {m1.matmul(m2)}")
print(f"shape of result: {(m1.matmul(m2)).shape}")  # (2, 1)

# 2. 행렬 곱셈 (3, 2) * (2, 1) => (3, 1)
print("-" * 30)
print("2. 행렬 곱셈 (3, 2) * (2, 1) => (3, 1)")

m1 = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
m2 = torch.FloatTensor([[1], [2]])  # (2, 1)
print(f"shape of m1: {m1.shape}, shape of m2: {m2.shape}")
print(f"m1.matmul(m2) 결과\n: {m1.matmul(m2)}")
print(f"shape of result: {(m1.matmul(m2)).shape}")  # (3, 1)

# 3. element-wise 곱셈 (동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱)
print("-" * 30)
print("3. element-wise 곱셈 (동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱)")

m1 = torch.FloatTensor([[1, 2], [3, 4]])  # (2, 2)
m2 = torch.FloatTensor([[1], [2]])  # (2, 1) => 브로드캐스팅 후 [[1, 1], [2, 2]]
print(f"shape of m1: {m1.shape}, shape of m2: {m2.shape}")
print(f"m1 * m2 결과\n: {m1 * m2}")  # 브로드캐스팅 적용
print(f"m1.mul(m2) 결과\n: {m1.mul(m2)}")  # 브로드캐스팅 적용
print(f"shape of result: {(m1 * m2).shape}")  # (2, 2)
