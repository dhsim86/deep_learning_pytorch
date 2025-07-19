import matplotlib.pyplot as plt

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
    plt.show()
