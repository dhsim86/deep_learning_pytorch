import numpy as np
import matplotlib.pyplot as plt


# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)

# W의 값이 커지면 경사가 커지고, 작아지면 경사가 작아진다.
plt.plot(x, y1, "r", linestyle="--")  # W의 값이 0.5일때
plt.plot(x, y2, "g")  # W의 값이 1일때
plt.plot(x, y3, "b", linestyle="--")  # W의 값이 2일때

plt.plot([0, 0], [1.0, 0.0], ":")  # 가운데 점선 추가
plt.title("Sigmoid Function")
plt.show()