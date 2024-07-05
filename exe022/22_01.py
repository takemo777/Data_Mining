from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。")

for token in tokens:
  print(token)
