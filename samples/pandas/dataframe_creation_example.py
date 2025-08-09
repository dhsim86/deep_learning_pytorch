import pandas as pd

# DataFrame 생성
# -> List, Series, dict, Numpy의 ndarray 등 다양한 자료형으로부터 생성 가능

# 1. List로부터 생성
data = [
    ["1000", "Steve", 90.72],
    ["1001", "James", 78.09],
    ["1002", "Doyeon", 98.43],
    ["1003", "Jane", 64.19],
    ["1004", "Pilwoong", 81.30],
    ["1005", "Tony", 99.14],
]
df = pd.DataFrame(data)  # 행과 열 인덱스는 0부터 시작하는 숫자로 생성됨
print(df)

df = pd.DataFrame(data, columns=["학번", "이름", "점수"])  # 열 이름 지정
print(df)

df = pd.DataFrame(data, columns=["학번", "이름", "점수"], index=["a", "b", "c", "d", "e", "f"])  # 행 인덱스 지정
print(df)

print("-" * 30)

# 2. Dict로부터 생성
data = {
    "학번": ["1000", "1001", "1002", "1003", "1004", "1005"],
    "이름": ["Steve", "James", "Doyeon", "Jane", "Pilwoong", "Tony"],
    "점수": [90.72, 78.09, 98.43, 64.19, 81.30, 99.14],
}
df = pd.DataFrame(data)  # 열 이름은 dict의 키로 지정됨
print(df)

# DataFrame 조회
print(df.head(3))  # 처음 3개 행 조회
print(df.tail(3))  # 마지막 3개 행 조회
print(df["학번"])  # 특정 열 ('학번') 조회

print(df.iloc[0])  # 첫 번째 행 조회

print("-" * 30)

# 3. 외부 데이터 (csv, jsonl 등)로부터 DataFrame 생성
# CSV 파일로부터 DataFrame 생성
df = pd.read_csv("scripts/pandas/example.csv")  # CSV 파일 경로 지정
print(df)
print(df.index)  # 인덱스는 자동으로 부여됨 (0부터 시작)
print(df.columns)  # 열 이름은 csv의 첫 번째 행으로부터 지정