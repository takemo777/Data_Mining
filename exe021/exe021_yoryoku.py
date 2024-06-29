import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# データの読み込み
df = pd.read_csv("exe21.csv")

x = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()

# 平均二乗誤差を計算する関数
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 多項式回帰と誤差計算の関数
def polynomial_regression(deg, x, y):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_pred = model.predict(x_poly)
    mse = mean_squared_error(y, y_pred)
    
    return y_pred, mse, model

least_squares_error = 10.5
deg = 1

# 最初の平均二乗誤差と予測値
y_pred, mse, model = polynomial_regression(deg, x, y)

# 多項式次数を増やす
while mse > least_squares_error:
    deg += 1
    y_pred, mse, model = polynomial_regression(deg, x, y)

print(f"平均二乗誤差: {mse}")
print(f"必要な最小多項式次数: {deg}")

plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x, y, label="データポイント")
plt.plot(x, y_pred, color="red", label="予測曲線")
plt.legend()
plt.show()

