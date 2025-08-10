import torch


ft = torch.FloatTensor([[0], [1], [2]])
print(f"원본 텐서:\n{ft} / shape: {ft.shape}") # (3, 1)

# squeeze, 1인 차원을 제거
print(f"squeeze 적용:\n{ft.squeeze()} / shape: {ft.squeeze().shape}") # (3, 1) => (3,)

print("-" * 30)

# unsqueeze, 1인 차원을 추가
ft = torch.Tensor([0, 1, 2])
print(f"원본 텐서:\n{ft} / shape: {ft.shape}") # (3,)

# 첫번째 차원에 1인 차원을 추가
print(f"unsqueeze 적용:\n{ft.unsqueeze(0)} / shape: {ft.unsqueeze(0).shape}") # (3,) => (1, 3)
print(f"-- view로도 가능:\n{ft.view(1, -1)} / shape: {ft.view(1, -1).shape}") # (3,) => (1, 3)
print(f"unsqueeze 적용:\n{ft.unsqueeze(1)} / shape: {ft.unsqueeze(1).shape}") # (3,) => (3, 1)