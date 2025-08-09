import pandas as pd

# pandas는 총 3가지 데이터 구조를 제공한다.
# Series, DataFrame, Panel

##### DataFrame
# DataFrame은 2차원 배열로부터 생성
# 열, 인덱스, 값으로 구성
# 행방향 인덱스와 열방향 인덱스가 존재
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 2차원 배열로부터 값을 생성
index = ["one", "two", "three"]
columns = ["A", "B", "C"]

df = pd.DataFrame(values, index=index, columns=columns)
print("\nDataFrame 출력:")
print("-" * 15)
print(df)

# 값과 열, 인덱스 출력
print(f"DataFrame의 인덱스: {df.index}")
print(f"DataFrame의 열: {df.columns}")
print(f"DataFrame의 값:\n{df.values}")

# DataFrame의 특정 열 선택
print(f"A 열 선택 :\n{df['A']}")  # 열 'A' 선택

# DataFrame의 특정 행 선택
print(f"'one' 행 선택 :\n{df.loc['one']}")  # 인덱스 'one' 선택
print(f"두 번째 행 선택 :\n{df.iloc[1]}")  # 인덱스 위치 1 선택
