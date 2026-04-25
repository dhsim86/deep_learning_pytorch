# Numpy를 활용하여 코사인 유사도 계산하는 함수를 구현 후 문서 벡터 간의 코사인 유사도 구하기
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

# 문서 1과 문서2의 유사도 : 0.6666666666666667
# 문서 1과 문서3의 유사도 : 0.6666666666666667
# 문서 2와 문서3의 유사도 : 1.0000000000000002
print('문서 1과 문서2의 유사도 :',cos_sim(doc1, doc2))
print('문서 1과 문서3의 유사도 :',cos_sim(doc1, doc3))
print('문서 2와 문서3의 유사도 :',cos_sim(doc2, doc3))