from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "James is working at Disney in London"
# 토큰화 후 품사 태깅
tokenized_sentence = pos_tag(word_tokenize(sentence))

# [('James', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), ('at', 'IN'), ('Disney', 'NNP'), ('in', 'IN'), ('London', 'NNP')]
print(tokenized_sentence)

# 개체명 인식
import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

ner_sentence = ne_chunk(tokenized_sentence)

#(S
#  (PERSON James/NNP)
#  is/VBZ
#  working/VBG
#  at/IN
#  (ORGANIZATION Disney/NNP)
#  in/IN
#  (GPE London/NNP))
print(ner_sentence)
