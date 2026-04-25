# 자카드 유사도 테스트

doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# 문서1 : ['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
# 문서2 : ['apple', 'banana', 'coupon', 'passport', 'love', 'you']
print('문서1 :',tokenized_doc1)
print('문서2 :',tokenized_doc2)

union = set(tokenized_doc1).union(set(tokenized_doc2))
# 문서1과 문서2의 합집합 : {'apple', 'passport', 'love', 'you', 'watch', 'coupon', 'everyone', 'like', 'card', 'holder', 'banana', 'likey'}
print('문서1과 문서2의 합집합 :',union)

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
# 문서1과 문서2의 교집합 : {'apple', 'banana'}
print('문서1과 문서2의 교집합 :',intersection)

# 자카드 유사도 : 0.16666666666666666
print('자카드 유사도 :',len(intersection)/len(union))