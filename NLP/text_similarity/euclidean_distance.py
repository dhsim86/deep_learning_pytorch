import numpy as np

# 유클리드 거리 계산 함수
def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


doc1 = np.array((2, 3, 0, 1))
doc2 = np.array((1, 2, 3, 1))
doc3 = np.array((2, 1, 2, 2))
docQ = np.array((1, 1, 0, 1))

# 문서1과 문서Q의 거리 : 2.23606797749979
# 문서2과 문서Q의 거리 : 3.1622776601683795
# 문서3과 문서Q의 거리 : 2.449489742783178
print("문서1과 문서Q의 거리 :", dist(doc1, docQ))
print("문서2과 문서Q의 거리 :", dist(doc2, docQ))
print("문서3과 문서Q의 거리 :", dist(doc3, docQ))
