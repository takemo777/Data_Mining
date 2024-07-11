from wordcloud import WordCloud
import cv2
import re

from janome.tokenizer import Tokenizer

unwanted_words = ['ベイダー', 'ルーク', 'スターウォーズ']

def wakachigaki(text, unwanted_words):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split('\t|,', str(line))
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
            if tkn[0] not in unwanted_words:
                words.append(tkn[0])
    return ' ' . join(words)

text = open("darth.txt", encoding="utf8").read()
text = wakachigaki(text, unwanted_words)
wordcloud = WordCloud(max_font_size=400,width=900,height=600,font_path='C:/Windows/Fonts/HGRSGU.TTC').generate(text)
wordcloud.to_file("result.png")

img = cv2.imread("result.png")
cv2.imshow("Image", img)
cv2.waitKey()
