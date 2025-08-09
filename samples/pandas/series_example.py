import pandas as pd

# pandas는 총 3가지 데이터 구조를 제공한다.
# Series, DataFrame, Panel

if __name__ == "__main__":
    ##### Series
    # Series는 1차원 배열로, 각 값에 대응되는 인덱스를 부여할 수 있는 구조이다.
    sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
    print("Series 출력:")
    print("-" * 15)
    print(sr)

    # 값과 인덱스 출력
    print(f"시리즈의 값: {sr.values}")
    print(f"시리즈의 인덱스: {sr.index}")

    print(sr["피자"])   # 피자의 값 출력
    print(sr.iloc[0])   # 첫 번째 값 출력
