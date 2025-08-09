import numpy as np

# 다차원 행렬 구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에 사용
# Numpy를 쓸 경우 편의성뿐만 아니라 속도면에서도 순수 파이썬에 비해 압도적으로 빠름

if __name__ == "__main__":
    # nparray는 리스트, 튜플, 배열로부터 생성 가능
    # 1차원이든 2차원이든 생성되는 타입은 ndarray이다.

    # 1차원 배열 생성
    vec = np.array([1, 2, 3, 4, 5])
    print(f"1차원 배열 (벡터) 출력: {vec}")
    print(f"vec의 타입: {type(vec)}")
    print(f"vec의 ndim: {vec.ndim}")    # 차원 수
    print(f"vec의 shape: {vec.shape}")  # 배열의 크기

    print("-" * 30)

    # 2차원 배열 생성
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"2차원 배열 (행렬) 출력:\n{mat}")
    print(f"mat의 타입: {type(mat)}")
    print(f"mat의 ndim: {mat.ndim}")    # 차원 수
    print(f"mat의 shape: {mat.shape}")  # 배열의 크기
