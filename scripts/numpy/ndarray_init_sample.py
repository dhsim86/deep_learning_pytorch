import numpy as np

# 다차원 행렬 구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에 사용
# Numpy를 쓸 경우 편의성뿐만 아니라 속도면에서도 순수 파이썬에 비해 압도적으로 빠름

if __name__ == "__main__":
    # ndarray 초기화

    # 1. zero로 초기화
    zero_mat = np.zeros((2, 3))  # 2x3 행렬을 0으로 초기화
    print(f"0으로 초기화된 행렬:\n{zero_mat}")
    
    print("-" * 30)

    # 2. 1로 초기화
    one_mat = np.ones((2, 3))  # 2x3 행렬을 1로 초기화
    print(f"1로 초기화된 행렬:\n{one_mat}")

    print("-" * 30)

    # 3. 특정 값으로 초기화
    fill_mat = np.full((2, 3), 7)  # 2x3 행렬을 7로 초기화
    print(f"7로 초기화된 행렬:\n{fill_mat}")

    print("-" * 30)

    # 4. 항등 행렬 (대각선의 원소가 모두 1이고 나머지는 0인 정사각형 행렬)
    identity_mat = np.identity(3)  # 3x3 항등 행렬 생성, np.eye(3, k=0)와 동일
    print(f"3x3 항등 행렬:\n{identity_mat}")
    # eye로도 항등 행렬 생성 가능
    eye_mat = np.eye(3)  # 3x3 항등 행렬
    print(f"3x3 eye 행렬:\n{eye_mat}")
    # eye는 값이 1인 대각선의 위치를 조절 가능
    eye_offset_mat = np.eye(4, k=1)  # 4x4 행렬에서 주대각선 1칸 위에 1을 위치시킴
    print(f"4x4 eye 행렬 (k=1):\n{eye_offset_mat}")

    print("-" * 30)

    # 5. 랜덤 값으로 초기화
    random_mat = np.random.rand(2, 3)  # 0과 1 사이의 랜덤 값으로 2x3 행렬 생성
    print(f"랜덤 값으로 초기화된 행렬:\n{random_mat}")

    print("-" * 30)

    # 6. np.arange로 초기화
    range_vec = np.arange(10)  # 0부터 9까지의 값으로 초기화된 1차원 배열
    print(f"np.arange로 초기화된 1차원 배열:\n{range_vec}")

    # 7. np.arange로 2씩 증가하는 배열 생성
    range_step_vec = np.arange(1, 10, 2)  # 1부터 9까지 2씩 증가
    print(f"np.arange로 2씩 증가하는 배열:\n{range_step_vec}")

    print("-" * 30)

    # 8. np.reshape로 배열의 형태 변경
    # 1차원 배열을 5x6 형태로 변경
    reshape_vec = np.arange(30)  # 0부터 29까지의 값으로 초기화된 1차원 배열
    reshape_mat = reshape_vec.reshape((5, 6))  # 5x6 형태로 변경
    print(f"reshape된 5x6 행렬:\n{reshape_mat}")

    print("-" * 30)

    # 9. 슬라이싱, 파이썬의 리스트처럼 슬라이싱 지원
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"원본 행렬:\n{mat}")
    sliced_row = mat[0, :]  # 첫 번째 행 전체 선택
    print(f"첫 번째 행 슬라이싱:\n{sliced_row}")
    sliced_col = mat[:, 1]  # 두 번째 열 전체 선택
    print(f"두 번째 열 슬라이싱:\n{sliced_col}")

    print("-" * 30)

    # 10. 정수 인덱싱, 연속적이지 않은 원소들로부터 배열을 만들 수 있음
    mat = np.array([[1, 2], [4, 5], [7, 8]])
    print(f"원본 행렬:\n{mat}")
    print(f"두 번째 행, 첫 번째 열의 원소 선택: {mat[1, 0]}")
    print(f"두 번째 행의 첫 번째 열과 첫 번째 행의 두 번째 열 선택: {mat[[1, 0], [0, 1]]}")