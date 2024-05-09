import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# データの読み込み
df = pd.read_csv("group_sample.csv")
X = df[['x1', 'x2']].to_numpy()
y_label = df['y'].to_numpy()

# 正解率を計算する関数
def calculate_accuracy(true_labels, cluster_labels):
    labels = np.unique(cluster_labels)
    pred_labels = np.empty_like(cluster_labels)
    
    for label in labels:
        mask = (cluster_labels == label)
        most_common = mode(true_labels[mask])[0][0]
        pred_labels[mask] = most_common
    
    return accuracy_score(true_labels, pred_labels)

# パラメータ範囲の設定
eps_range = np.arange(0.1, 3.1, 0.1)
min_samples_range = range(2, 21)

accuracy_with_noise = 0
accuracy_without_noise = 0
params_with_noise = {}
params_without_noise = {}

# グリッドサーチ
for eps in eps_range:
    for min_samples in min_samples_range:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        
        # 全クラスタ数（ノイズ含む）
        n_clusters_with_noise = len(set(model.labels_))
        
        # ノイズを除いたクラスタ数
        if -1 in model.labels_:
            n_clusters_without_noise = len(set(model.labels_)) - 1
        else:
            n_clusters_without_noise = len(set(model.labels_))

        # ノイズを含む場合
        if n_clusters_with_noise == 3:
            accuracy = calculate_accuracy(y_label, model.labels_)
            if accuracy > accuracy_with_noise:
                accuracy_with_noise = accuracy
                params_with_noise = {"eps": eps, "min_samples": min_samples}
        
        # ノイズを含まない場合
        if n_clusters_without_noise == 3:
            accuracy = calculate_accuracy(y_label, model.labels_)
            if accuracy > accuracy_without_noise:
                accuracy_without_noise = accuracy
                params_without_noise = {"eps": eps, "min_samples": min_samples}

# 結果の表示
print(f"ノイズを含む場合の正解率: {accuracy_with_noise * 100}%, パラメータ: {params_with_noise}")
print(f"ノイズを含まない場合の正解率: {accuracy_without_noise * 100}%, パラメータ: {params_without_noise}")
