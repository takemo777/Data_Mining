import re

test_str ="<html><body>AAAAAA</body></html>"

result = re.sub('<.+>','BBB',test_str)
#result = re.sub('<.+?>','BBB',test_str)
if result:
    print(result)
    print("マッチしました")
else:
    print("マッチしてません")
#print(type(result))