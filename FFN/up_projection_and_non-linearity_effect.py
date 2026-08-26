# FFN 레이어에서 차원을 확장하는 Up-projection과 이후 적용되는 비선형 활성화 함수는 고차원 공간에서의 풍부한 특징 표현이 가능
# 8차원의 작은 모델은 데이터의 복잡한 패턴을 포착하기 어려워 높은 손실값을 보이는 반면, 
# 64차원의 큰 모델은 비선형 패턴을 더 정확하게 포착하여 예측의 정확도가 크게 향상

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate data: complex non-linear function
np.random.seed(42)
X = np.linspace(-1, 1, 1000)
y = X**3 + 0.1 * np.random.normal(size=X.shape)

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X.reshape(-1, 1)).float()
y_tensor = torch.from_numpy(y.reshape(-1, 1)).float()

# Split data into training and test sets
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Model 1: Small intermediate dimension (limited expressiveness)
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Model 2: Large intermediate dimension (increased expressiveness)
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Initialize models
model_small = SmallModel()
model_large = LargeModel()

# Define loss function and optimizers
criterion = nn.MSELoss()
optimizer_small = optim.Adam(model_small.parameters(), lr=0.01)
optimizer_large = optim.Adam(model_large.parameters(), lr=0.01)

# Training function definition
def train_model(model, optimizer, X_train, y_train, epochs=1000):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Train models
model_small = train_model(model_small, optimizer_small, X_train, y_train)
model_large = train_model(model_large, optimizer_large, X_train, y_train)

# Evaluate models
model_small.eval()
model_large.eval()
with torch.no_grad():
    y_pred_small = model_small(X_test)
    y_pred_large = model_large(X_test)
    loss_small = criterion(y_pred_small, y_test).item()
    loss_large = criterion(y_pred_large, y_test).item()

print(f"Small Model Test Loss: {loss_small:.4f}")
print(f"Large Model Test Loss: {loss_large:.4f}")

# Visualization
X_test_np = X_test.numpy()
y_test_np = y_test.numpy()
y_pred_small_np = y_pred_small.numpy()
y_pred_large_np = y_pred_large.numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_np, y_test_np, label='Actual', alpha=0.5)
plt.scatter(X_test_np, y_pred_small_np, label='Predicted (Small Model)', alpha=0.5)
plt.title('Small Model Predictions')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test_np, y_test_np, label='Actual', alpha=0.5)
plt.scatter(X_test_np, y_pred_large_np, label='Predicted (Large Model)', alpha=0.5)
plt.title('Large Model Predictions')
plt.legend()

plt.show()