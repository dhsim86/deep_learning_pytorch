import numpy as np

if __name__ == '__main__':
    # 스칼라 연산, 즉 요소끼리 그냥 곱하는 연산
    mat1 = np.array([1, 2, 3])
    mat2 = np.array([4, 5, 6])  # 두 개의 1차원 배열

    # 덧셈
    # np.add(mat1, mat2)와 동일
    print(f"두 개의 1차원 배열 덧셈 결과:\n{ mat1 + mat2}")

    # 뺄셈
    # np.subtract(mat1, mat2)와 동일
    print(f"두 개의 1차원 배열 뺄셈 결과:\n{ mat1 - mat2}")

    # 곱셈
    # np.multiply(mat1, mat2)와 동일
    print(f"두 개의 1차원 배열 곱셈 결과:\n{ mat1 * mat2}")

    # 나눗셈
    # np.divide(mat1, mat2)와 동일
    print(f"두 개의 1차원 배열 나눗셈 결과:\n{ mat1 / mat2}")
    
    print("-" * 30)

    # 행렬 곱
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])  # 2x2 행렬
    print(f"행렬 곱 결과:\n{np.dot(mat1, mat2)}")  # np.matmul(mat1, mat2)와 동일