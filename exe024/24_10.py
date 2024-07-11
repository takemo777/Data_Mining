import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split('\t|,', str(line))
        words.append(tkn[0])
    return ' ' . join(words)

text = open("soseki.txt", encoding="utf8").read()
text = wakachigaki(text)
print(text)

