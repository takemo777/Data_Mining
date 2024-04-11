import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

"""(1) この CSV を変数名 df というデータフレームに読み込みなさい。
	ただし、A_id 列は削除すること。"""
# CSVファイルの読み込み
df = pd.read_csv("apple_quality.csv")

# A_id列の削除
df.drop(columns=["A_id"], axis = 1)
#print(df)

"""(2) Size 列,  Weight 列の最小値、最大値、平均値を求めなさい。"""
# Size列の最小値、最大値、平均値を計算
size_min = df['Size'].min()
size_max = df['Size'].max()
size_mean = df['Size'].mean()

# Weight列の最小値、最大値、平均値を計算
weight_min = df['Weight'].min()
weight_max = df['Weight'].max()
weight_mean = df['Weight'].mean()

# 結果の表示
print(f"Sizeの最小値、最大値、平均値は {size_min}, {size_max}, {size_mean}")
print(f"Weightの最小値、最大値、平均値は {weight_min}, {weight_max}, {weight_mean}")

"""(3) データフレーム df の先頭 1200 行のデータを取り出し、NumPy配列にしなさい。"""
df_first_1200 = df.iloc[:1200].to_numpy()
#print(df_first_1200)

"""(4）（3）の 1200 行のデータを用い、横軸に Size を、縦軸に Weight をとって、散布図を描きなさい。（目的：Matplotlib でグラフで単純な散布図を描くことができるか。）"""
# SizeとWeightの列をNumPy配列から抽出
size_np = df_first_1200[:, 0]
weight_np = df_first_1200[:, 1]

# 散布図を描画
plt.figure(figsize=(10, 6))
# 透明度を調整して表示
plt.scatter(size_np, weight_np, color = "red")
# グラフのタイトル
plt.title("散布図")
# X軸のラベル
plt.xlabel("Size")
# Y軸のラベル
plt.ylabel("Weight")
# グリッドを表示
plt.grid(True)
# グラフを表示
plt.show()

"""（5）Quality列の値は文字列であるが、２種の値になっているかを確認せよ。また、全データについて、good は 1 に、badは 0 にラベルエンコーディングしなさい。"""
unique_values = df['Quality'].unique()

# ラベルエンコーディング: goodを1に、badを0に変換
df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})
print(unique_values)
# ↑つまり２種の値
#print(df['Quality'])

"""（6）Size,  Weight,  Sweetness,  Crunchiness,  Juiciness,  Ripeness,Acidity の７パラメータから Quality を予測する分類モデルを作りなさい。訓練データとして先頭の1200行のデータを使いなさい。その際、scikit-learn を使い、どんな学習モデルを使うかは各自で決めなさい。"""
# 特徴量とターゲットを分ける
X = df.drop('Quality', axis=1)
y = df['Quality']

# 訓練データとして先頭1200行を使用し、残りをテストデータとする
X_train = X.iloc[:1200]
y_train = y.iloc[:1200]
X_test = X.iloc[1200:]
y_test = y.iloc[1200:]

# モデルのインスタンス作成
#model = LogisticRegression(max_iter=1000, random_state=50)
model = ExtraTreesClassifier()
#model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#                           beta_2=0.999, early_stopping=False, epsilon=1e-08,
#                           hidden_layer_sizes=(30,30,30), max_iter=10000)

# モデルの訓練
model.fit(X_train, y_train)

# 訓練データとテストデータの予測値を計算
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 訓練データとテストデータの精度を計算
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"訓練データ：{train_accuracy * 100}%")
print(f"テストデータ；{test_accuracy * 100}%")
