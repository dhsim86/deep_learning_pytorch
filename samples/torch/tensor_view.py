import torch
import numpy as np

t = np.array([[[0, 1, 2], [3, 4, 5]], 
              [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t) # (2, 2, 3) 형태의 텐서로 변환

print(f"origin tensor(2, 2, 3):\n{ft}")
print(f"origin tensor shape: {ft.shape}")

print("-" * 30)

# 원소 갯수는 유지, pytorch는 사이즈가 -1로 설정시 다른 차원으로부터 값을 유추
# 1. 2차원 텐서로 변경
print(f"2차원 텐서로 변경:\n{ft.view([-1, 3])}") # (2, 2, 3) => (?, 3) => (2 * 2, 3) => (4, 3)
print(f"2차원 텐서 shape: {ft.view([-1, 3]).shape}")

print("-" * 30)

# 2. 3차원 텐서로 유지하되 크기만 변경
print(f"3차원 텐서로 유지하되 크기만 변경:\n{ft.view([-1, 1, 3])}") # (2, 2, 3) => (4, 1, 3)
print(f"3차원 텐서 shape: {ft.view([-1, 1, 3]).shape}")
