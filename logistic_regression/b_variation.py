import numpy as np
import matplotlib.pyplot as plt


# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x + 0.5)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 1.5)

plt.plot(x, y1, "r", linestyle="--")  # x + 0.5
plt.plot(x, y2, "g")  # x + 1
plt.plot(x, y3, "b", linestyle="--")  # x + 1.5

plt.plot([0, 0], [1.0, 0.0], ":")  # 가운데 점선 추가
plt.title("Sigmoid Function")
plt.show()
