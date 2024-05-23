import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import japanize_matplotlib

# Irisデータセットの読み込み
df = pd.read_csv("iris.csv", sep=",")
df = df.sample(frac=1)  # データのシャッフル
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
labels = df["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2}).to_numpy()

rates = [0.03, 0.05, 0.08]
results = []

def comp_model(X, labels, t_n):
    # ラベル拡散モデルの訓練と予測
    model1 = LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=7, max_iter=1000, n_jobs=-1)
    model1.fit(X, labels)
    result1 = model1.predict(X)
    
    # SVMモデルの訓練と予測
    model2 = SVC(gamma="scale")
    model2.fit(X[t_n:, :], labels[t_n:])
    result2 = model2.predict(X)
    
    return [result1, result2]

def accuracy_array(array_a, array_b):
    array_c = (array_a == array_b)
    return int(np.count_nonzero(array_c) / len(array_c) * 100)

for rate in rates:
    t_n = int(len(X) * (1 - rate))

    labels_org = labels.copy()
    labels_partial = labels.copy()
    labels_partial[0:t_n] = np.array([-1] * t_n)

    rest1, rest2 = comp_model(X, labels_partial, t_n)
    
    results.append({
        "割合": rate,
        "ラベル拡散": accuracy_array(rest1, labels_org),
        "SVM": accuracy_array(rest2, labels_org)
    })

# 結果の表示
df_results = pd.DataFrame(results)
print(df_results)

# 散布図
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']
labels_names = ['setosa', 'versicolor', 'virginica']

for i, color, marker, label_name in zip(range(3), colors, markers, labels_names):
    plt.scatter(X[labels == i, 2], X[labels == i, 3], c=color, marker=marker, label=label_name)

plt.xlabel('花弁の長さ')
plt.ylabel('花弁の幅')
plt.legend()
plt.title('散布図（花弁の長さと花弁の幅）')
plt.grid(True)
plt.show()
