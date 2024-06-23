import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import PredictionErrorDisplay

df = pd.read_csv("Transistor_count.csv")

df["log_transistor_count"] = np.log10(df["MOS_transistor_count"])

"""（1）xy 平面にこのデータを散布図としてプロットせよ。"""
plt.scatter(df['year'], df['MOS_transistor_count'])
plt.xlabel('year')
plt.ylabel('MOS_transistor_count')
plt.title('CPUのトランジスタ数')
plt.legend()
plt.grid(True)
plt.savefig("mini017(1).png")
plt.show()


"""（2）次に、縦軸を log10(y)（常用対数）としてプロットしてみよ"""
# データの整形
x = df["year"].values.reshape(-1, 1)
y = df["log_transistor_count"].values

# モデルの訓練
model = LinearRegression()
model.fit(x, y)

# 回帰直線の予測値を計算
y_pred = model.predict(x)

plt.scatter(df["year"], df["log_transistor_count"])
plt.plot(df["year"], y_pred, color='red')
plt.xlabel("Year")
plt.ylabel("Log10(MOS Transistor Count)")
plt.title("CPUのトランジスタ数 (常用対数)")
plt.legend()
plt.grid(True)
plt.savefig("mini017(2).png")
plt.show()

"""（3）z = log10(y) として、x から z を予測する式を求めよ。"""

coef = model.coef_[0]
intercept = model.intercept_

print(f"予測式: z = {coef:.4f} * x + {intercept:.4f}")

"""（4）x から y を予測する式を求めよ。"""

def predict_transistor_count(year):
    z = coef * year + intercept
    return 10 ** z

print(f"y = 10^({coef:.4f} * x + {intercept:.4f})")

y2022 = predict_transistor_count(2022)
y2020 = predict_transistor_count(2020)
rate = y2022 / y2020
double_2020 = 2 * y2020
print(f"2020年のトランジスタ数: {y2020:.4f} 2022年のトランジスタ数: {y2022:.4f} 2020年のトランジスタ数の2倍:{double_2020:.4f}" )
print(f"2022/2020の比率:{rate:.4f}")

"""（5）残差を残差プロットに表せ。（x, z の回帰で）"""

Residual = y - y_pred

plt.scatter(df["year"], Residual)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Residual")
plt.title("残差プロット")
plt.grid(True)
plt.savefig("mini017(5).png")
plt.show()

"""（6）x, z の回帰で R2 決定係数を求めよ。"""

predicted_z = model.predict(x)
r2 = r2_score(y, predicted_z)
print(f"決定係数 R2: {r2:.4f}")

"""（7）この結果から、ムーアの法則（yは、2年ごとに2倍になる）は正しいといえるかどうかあなたの考えを述べなさい。"""
