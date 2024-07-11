import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = {}
    for line in token:
        tkn = re.split('\t|,', str(line))
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
            words[tkn[0]] = words[tkn[0]] +1 if tkn[0] in words else 1
    words = sorted(words.items(), key=lambda x:x[1])
    return words

text = open("soseki.txt", encoding="utf8").read()
text_list = wakachigaki(text)
print(text_list)