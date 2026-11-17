# 확장된 차원에서의 비선형 변환이 강화
# 비선형 현상에 대한 모델링 향상

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

# Generate data: two moon shapes dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Model 1: Small intermediate dimension
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out

# Model 2: Large intermediate dimension (enhanced non-linearity)
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize models
model_small = SmallModel()
model_large = LargeModel()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_small = optim.Adam(model_small.parameters(), lr=0.01)
optimizer_large = optim.Adam(model_large.parameters(), lr=0.01)

# Train function
def train_model(model, optimizer, X, y, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model

model_small = train_model(model_small, optimizer_small, X_tensor, y_tensor)
model_large = train_model(model_large, optimizer_large, X_tensor, y_tensor)

# Evaluate models
model_small.eval()
model_large.eval()
with torch.no_grad():
    outputs_small = model_small(X_tensor)
    outputs_large = model_large(X_tensor)
    preds_small = (outputs_small > 0.5).float()
    preds_large = (outputs_large > 0.5).float()
    acc_small = (preds_small == y_tensor).float().mean().item()
    acc_large = (preds_large == y_tensor).float().mean().item()

print(f"Small Model Accuracy: {acc_small:.4f}")
print(f"Large Model Accuracy: {acc_large:.4f}")

# Decision boundary visualization
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid.astype(np.float32))
    with torch.no_grad():
        Z = model(grid_tensor)
    Z = Z.reshape(xx.shape)
    Z = Z > 0.5

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.show()

# Visualization
plot_decision_boundary(model_small, X, y, 'Small Model Decision Boundary')
plot_decision_boundary(model_large, X, y, 'Large Model Decision Boundary')