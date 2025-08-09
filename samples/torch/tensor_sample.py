import torch

# 1. 1차원 텐서(벡터) 생성
t = torch.FloatTensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print("1차원 텐서 (벡터):", t)
print(f"텐서의 차원: {t.dim()}")
print(f"텐서의 형태: {t.shape}")
print(f"텐서의 크기: {t.size()}")

print(f"인덱스로 접근: {t[0]}")  # 첫 번째 요소
print(f"슬라이싱: {t[1:4]}")  # 1부터 3까지의 요소

print("-" * 30)

# 2. 2차원 텐서(행렬) 생성
t = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
print("2차원 텐서 (행렬):\n", t)

print(f"텐서의 차원: {t.dim()}")
print(f"텐서의 형태: {t.shape}")
print(f"텐서의 크기: {t.size()}")

print(f"인덱스로 접근: {t[0]}")  # 첫 번째 행
print(f"슬라이싱: {t[1:3]}")  # 1부터 2까지의 행

print(f"2번째 열: {t[:, 1]}")  # 2번째 열 선택
print(f"2번째 열의 크기: {t[:, 1].size()}")

print(f"마지막 열을 제외한 텐서:\n{t[:, :-1]}")
