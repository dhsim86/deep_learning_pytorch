import matplotlib.pyplot as plt

import numpy as np

# 2차 함수 그래프 그리기 함수
def plot_quadratic(a, b, c, x_range=(-10, 10), num_points=100):
    """
    2차 함수 y = ax^2 + bx + c 그래프를 그립니다.
    Args:
        a, b, c: 2차 함수 계수
        x_range: x축 범위 (튜플)
        num_points: x축 샘플 개수
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = a * x ** 2 + b * x + c
    plt.figure()
    plt.plot(x, y, label=f"y = {a}x² + {b}x + {c}")
    plt.title("Quadratic Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 라인 플롯
    plt.title("Line Plot Example")
    plt.plot([1, 2, 3, 4], [2, 4, 8, 6])  # x축과 y축 데이터
    plt.xlabel("hours")  # x축 레이블
    plt.ylabel("score")  # y축 레이블
    # plt.show()

    # 여러개의 라인 플롯
    plt.cla()  # 이전 플롯 지우기
    plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
    plt.plot([1.5, 2.5, 3.5, 4.5], [3, 5, 8, 10])  # 라인 새로 추가
    plt.xlabel("hours")
    plt.ylabel("score")
    plt.legend(["A student", "B student"])  # 범례 삽입
    # plt.show()

    plot_quadratic(1, 0, 0)