# 차원을 축소하면서 중요한 정보만 선별적으로 유지하는 병목 효과가 발생
# 입력에 있는 불필요한 노이즈를 자연스럽게 제거하고, 핵심적인 특징만이 압축
# 모델의 과적합을 방지하는 정규화 효과도 함께 가져오며, 모델의 일반화 능력을 향상

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate data: noisy sine wave signal
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(X)
y_noisy = y + 0.5 * np.random.normal(size=X.shape)

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X.reshape(-1, 1)).float()
y_noisy_tensor = torch.from_numpy(y_noisy.reshape(-1, 1)).float()
y_tensor = torch.from_numpy(y.reshape(-1, 1)).float()

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: Expansion
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Bottleneck
            nn.ReLU()
        )
        # Decoder: Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
autoencoder = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# Train model
epochs = 500
autoencoder.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = autoencoder(X_tensor)
    loss = criterion(outputs, y_noisy_tensor)
    loss.backward()
    optimizer.step()

# Predict and visualize
autoencoder.eval()
with torch.no_grad():
    y_denoised = autoencoder(X_tensor).numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(X, y_noisy, label='Noisy Signal')
plt.plot(X, y, label='Original Signal')
plt.title('Input Signal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X, y_denoised, label='Denoised Signal')
plt.plot(X, y, label='Original Signal')
plt.title('Denoised Output')
plt.legend()

plt.show()