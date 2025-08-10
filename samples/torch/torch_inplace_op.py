import torch

x = torch.FloatTensor([[1, 2], [3, 4]])

# 1. 일반 연산은 텐서가 in place 업데이트되지 않음
print(f"element-wise 곱셈:\n{x.mul(2.)}")
print(f"원본 텐서는 변하지 않음:\n{x}")

# 2. in place 연산은 텐서가 업데이트됨
print(f"in place element-wise 곱셈:\n{x.mul_(2.)}")
print(f"원본 텐서가 업데이트됨:\n{x}")