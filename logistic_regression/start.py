import numpy as np
import matplotlib.pyplot as plt


# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, "g")
plt.plot([0, 0], [1.0, 0.0], ":")  # 가운데 점선 추가
plt.title("Sigmoid Function")
plt.show()
