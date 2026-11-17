# 병목이 지나치면 필요한 특징까지 손실될 수 있음
# 중간 차원이 지나치게 작아지면 정보가 충분히 전달되지 않아 모델이 안정적으로 학습하지 못하고 높은 손실을 보일 수 있다.
# 충분한 중간 차원을 확보하면 정보의 원활한 흐름이 보장되어 학습 과정이 더 안정적이며 성능도 향상

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate data: XOR problem, a non-linear binary classification task
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y.reshape(-1, 1))

# Model 1: Small intermediate dimension (bottleneck)
class BottleneckModel(nn.Module):
    def __init__(self):
        super(BottleneckModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Small hidden layer causes bottleneck
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out



# Model 2: Sufficient intermediate dimension (prevents bottleneck)
class NoBottleneckModel(nn.Module):
    def __init__(self):
        super(NoBottleneckModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # Larger hidden layer for smoother flow
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize models
model_bottleneck = BottleneckModel()
model_no_bottleneck = NoBottleneckModel()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_bottleneck = optim.Adam(model_bottleneck.parameters(), lr=0.1)
optimizer_no_bottleneck = optim.Adam(model_no_bottleneck.parameters(), lr=0.1)

# Training function
def train_model(model, optimizer, X, y, epochs=500):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history

# Train both models
loss_history_bottleneck = train_model(model_bottleneck, optimizer_bottleneck, X_tensor, y_tensor)
loss_history_no_bottleneck = train_model(model_no_bottleneck, optimizer_no_bottleneck, X_tensor, y_tensor)

# Visualization of loss history
plt.figure(figsize=(12, 6))
plt.plot(loss_history_bottleneck, label='Bottleneck Model')
plt.plot(loss_history_no_bottleneck, label='No Bottleneck Model')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()