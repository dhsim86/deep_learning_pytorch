import torch

# 같은 크기일 때 연산
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(f"같은 크기 연산 결과: {m1 + m2}")

print("-" * 30)

# 크기가 다른 경우 (행렬 * 스칼라)
m1 = torch.FloatTensor([[1, 2,]])
m2 = torch.FloatTensor([3]) # --> [3, 3]로 브로드캐스팅
print(f"크기가 다른 경우 연산 결과(행렬(1, 2) * 스칼라(1,)): {m1 + m2}")

print("-" * 30)

# 크기가 다른 경우
m1 = torch.FloatTensor([[1, 2]])    # --> [[1, 2], [1, 2]], 행방향으로 확장
m2 = torch.FloatTensor([[3], [4]])  # --> [[3, 3], [4, 4]], 열방향으로 확장
print(f"크기가 다른 경우 연산 결과(행렬(1, 2) + 행렬(2, 1)):\n{m1 + m2}")