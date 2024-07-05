from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

tokenizer = Tokenizer()

def get_txt(file_name):
    with open(file_name, 'r' , encoding='UTF-8') as f:
        txt = f.read()
        txt = txt.replace("\n", "")
        
    return txt
    
def nou_waka(txt_t,mode="txt",only_noum=False):
    tokens = tokenizer.tokenize(txt_t,wakati=False)
    wakati_list = []
    
    for token in tokens:
        if only_noum:
            if token.part_of_speech.split(',')[0] =='名詞':
                wakati_list.append(token.surface)
        else:
            wakati_list.append(token.surface)        
    
    if mode=="txt":
        return " ".join(wakati_list)
    else:
        return wakati_list

def txt_to_list(txt):
    return_list = []
    for t in txt.split('。'):
        if t.strip():  # 空の文を取り除く
            return_list.append(nou_waka(t, mode="txt"))
    return return_list

txt = get_txt("sentence.txt")

print(txt_to_list(txt))

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(txt_to_list(txt))
print(vectorizer.get_feature_names_out())
print(X.toarray())
print(len(txt_to_list(txt)))
