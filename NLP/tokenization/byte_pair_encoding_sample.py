import re, collections
from IPython.display import display, Markdown, Latex

# BPE 몇번 수행할 것인지 지정
num_merges = 10

# BPE에 사용할 단어가 low, lower, newest, widest일 때, BPE의 입력으로 사용하는 실제 단어 집합
# </w>는 단어의 맨 끝에 붙이는 특수 문자이며, 각 단어는 글자(character) 단위로 분리
dictionary = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}

# BPE 코드, 가장 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정을 num_merges만큼 반복
def get_stats(dictionary):
    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    print("현재 pair들의 빈도수 :", dict(pairs))
    return pairs


def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


bpe_codes = {}
bpe_codes_reverse = {}

for i in range(num_merges):
    display(Markdown("### Iteration {}".format(i + 1)))
    pairs = get_stats(dictionary)
    best = max(pairs, key=pairs.get)
    dictionary = merge_dictionary(best, dictionary)

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best

    print("new merge: {}".format(best))
    print("dictionary: {}".format(dictionary))
