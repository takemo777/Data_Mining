from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import numpy as np

df = pd.read_csv('003.csv')
print("データフレーム概要を再確認")
print(df.info())

print("\n")
print("重複した行がないか確認します。")
print(df.duplicated().sum())

print("\n")
print("欠損値を各特徴量について確認します。")
print(df[df.isnull().any(axis = 1)])

X = df[['x1','x2']].to_numpy()

y_label = df['label'].to_numpy() #yは本当のグループだが、クラスタリングには使わない。

model = KMeans(n_clusters=2, random_state=0,n_init="auto")
y_clstr = model.fit_predict(X)
print(model.labels_)

#散布図をぽろっとして、クラスタごとに色分けする
plt.scatter(X[:,0],X[:,1],c=model.labels_)

plt.scatter(2.20,2.89,color="red")
plt.text(2.20,2.89, "A" )

plt.scatter(2.02,1.69,color="red")
plt.text(2.02,1.69, "B" )

plt.scatter(2.83,1.52,color="red")
plt.text(2.83,1.52, "C" )

plt.scatter(1.10,2.51,color="red")
plt.text(1.10,2.51, "D" )

plt.savefig("clustering_result.png")
plt.show()

#以下はS.Raschka他『Pythoh機械学習プログラミング　第３版』（インプレス）のコードを引用している
cluster_labels = np.unique(y_clstr) 
n_clusters=cluster_labels.shape[0] 

silhouette_vals = silhouette_samples(X,y_clstr,metric='euclidean')  # シルエット係数を計算
y_ax_lower, y_ax_upper= 0,0
yticks = []
bar_color=['#CC4959','#33cc33','#4433cc']
for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_clstr==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
        color = cm.jet(float(i)/n_clusters)               # 色の値を作る
        plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                         c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                         height=1.0,                      # 棒の高さ
                         edgecolor='none',                # 棒の端の色
                         #color=color)
                         color=bar_color[i])              # 棒の色
        yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
        y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
plt.ylabel('Cluster')
plt.xlabel('silhouette coefficient')
plt.savefig("silhouette_result.png")
plt.show()