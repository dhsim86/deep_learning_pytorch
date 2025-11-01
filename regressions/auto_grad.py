import torch

# 값이 2인 임의의 스칼라 w 생성, 학습을 통해 값이 변경되는 변수로 명시
# w.grad에 w에 대한 미분값이 저장됨
w = torch.tensor(2.0, requires_grad=True)

y = w ** 2 # y = w^2
z = 2 * y + 5 # z = 2y + 5

z.backward() # z를 w로 미분(∂z/∂w)하여 w.grad에 저장
print("자동 미분으로 계산된 w에 대한 z의 미분값:", w.grad) # dz/dw = 4w = 8.0
