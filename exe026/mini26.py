import re

# 練習問題(1)
"""7 桁の数字を 3 桁と 4 桁に分けて間を - で結んだ文字列に変換する関数conv3_4 をつくれ。
入力が 7 桁の数字でない場合は Error! という文字列を返せ。
"""

def conv3_4(str7:str)->str:
    pattern = r'(\d{3})(\d{4})'
    result = re.sub(pattern, r'\1-\2', str7)
    if result != str7:
        return result
    else:
        return "Error!"

print(conv3_4("1670031"))
print(conv3_4("7640003"))
print(conv3_4("123ABCD"))

# 練習問題(2)
"""
次のような数字 24 時間表記で書かれた時刻のデータがある。
このデータのうち、19:00 から 23:59 までの時刻の場合、
時間の部分の数字を文字列として取り出す関数get_19_24を書け。
それ以外の時間帯なら空文字を返せ。"""

def get_19_24(time_str: str) -> str:
    match = re.match(r'(\d{2}):(\d{2})', time_str)
    if match:
        hour, minute = match.groups()
        #hour = match.group(1)
        hour = int(hour)
        if 19 <= hour <= 23:
            return str(hour)
    return ''

print(get_19_24("15:21"))
print(get_19_24("19:11"))
print(get_19_24("23:34"))

# 練習問題(3)
"""GoodやGodのようにGで始まり、oが1個以上続いて、
d終わる文字列かどうかを判定する関数 is_god を書け。"""

def is_god(str1:str)->str:
    pattern = r'Go+d'
    result = re.match(pattern, str1)
    return True if result else False 
    
print(f"{is_god('God')}")
print(f"{is_god('Good')}")
print(f"{is_god('Goad')}")
print(f"{is_god('Goooooooooooooooooooooogle')}")
