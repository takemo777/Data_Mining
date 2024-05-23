import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import japanize_matplotlib

def is_zero(th, data, labels):
    # PCAで次元削減
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # t-SNEでさらに次元削減して視覚化
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(reduced_data)

    # クラスタの中心からの距離を計算して閾値で判定
    center = np.mean(tsne_result[labels == 0], axis=0)
    distances = np.linalg.norm(tsne_result - center, axis=1)

    # 0かそれ以外を判別
    result = distances < th

    # t-SNE結果の可視化
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))

    for i in range(10):
        indices = labels == i
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], c=[colors[i]], label=f'{i}', alpha=0.5)

    plt.scatter(center[0], center[1], c='red', marker='*', s=100, label='0のクラスタ中心')
    circle = plt.Circle(center, th, color='red', fill=False, linestyle='dashed', label=f'閾値 {th}')
    plt.gca().add_artist(circle)
    plt.title('手書き数字のt-SNE可視化')
    plt.legend()
    plt.show()

    return result

# データの読み込み
digits = datasets.load_digits()
digits_data = digits.data
digits_label = digits.target

# 閾値の設定
threshold = 14.0

# 判別結果の取得
predictions = is_zero(threshold, digits_data, digits_label)

# 正解率の計算
accuracy = accuracy_score(digits_label == 0, predictions)

print(f"正解率: {accuracy * 100:.2f}%")
