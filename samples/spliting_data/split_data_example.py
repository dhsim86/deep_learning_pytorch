import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# 1. zip 함수를 이용하여 데이터 분리
# zip() 함수는 동일한 갯수를 가지는 시퀀스 자료형에서 각 순서에 등장하는 요소들끼리 묶어준다.
X, y = zip(["a", 1], ["b", 2], ["c", 3])
print(f"x 데이터: {X}")
print(f"y 데이터: {y}")

X, y = zip(*[["a", 1], ["b", 2], ["c", 3]])  # 리스트 언패킹
print(f"x 데이터: {X}")
print(f"y 데이터: {y}")

print("-" * 30)

# 2. DataFrame을 이용하여 분리
values = [
    ["당신에게 드리는 마지막 혜택!", 1],
    ["내일 뵐 수 있을지 확인 부탁드...", 0],
    ["도연씨. 잘 지내시죠? 오랜만입...", 0],
    ["(광고) AI로 주가를 예측할 수 있다!", 1],
]
columns = ["메일 본문", "스팸 메일 유무"]

# DataFrame은 열의 이름으로 각 열에 접근 가능하므로 쉽게 분리 가능
df = pd.DataFrame(values, columns=columns)
print(f"DataFrame:\n{df}\n")
print(f"X 데이터:\n{df['메일 본문']}\n")
print(f"y 데이터:\n{df['스팸 메일 유무']}\n")

print("-" * 30)

# 3. numpy를 이용하여 분리, 슬라이싱을 사용
np_array = np.arange(0, 16).reshape(4, 4)
print(f"numpy 배열:\n{np_array}")

X = np_array[:, :3]  # 첫 3열
y = np_array[:, 3]  # 마지막 열
print(f"X 데이터:\n{X}")
print(f"y 데이터:\n{y}")

print("-" * 30)

# 4. train_test_split을 이용하여 분리
# 사이킷런은 학습용과 테스트용 데이터를 분리하는 함수인 train_test_split을 제공한다.
X = df["메일 본문"]
y = df["스팸 메일 유무"]

# X = 독립 변수 데이터
# y = 종속 변수 데이터 (레이블)
# test_size: 테스트 데이터의 갯수 (1보다 작은 실수이면 비율)
# random_state: 난수 생성 시드
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(f"X_train 데이터:\n{X_train}")
print(f"y_train 데이터:\n{y_train}")
print(f"X_test 데이터:\n{X_test}")
print(f"y_test 데이터:\n{y_test}")
