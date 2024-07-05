from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = Tokenizer()

def get_txt(file_name):
    with open(file_name, 'r' , encoding='UTF-8') as f:
        txt = f.read()
        txt = txt.replace("\n", "")
    return txt
    
def nou_waka(txt, mode="txt"):
    tokens = tokenizer.tokenize(txt)
    wakati_list = []
    for token in tokens:
        if token.part_of_speech.split(',')[0] == '名詞':
            wakati_list.append(token.surface)
    
    if mode == "txt":
        return " ".join(wakati_list)
    else :
        return wakati_list

txt = get_txt("sentence.txt")

print(nou_waka(txt))
