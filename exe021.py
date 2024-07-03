import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# データの読み込み
df = pd.read_csv("exe21.csv")

# 関数の定義
def func(x, A, B, C):
    return A * x + B * np.sin(2 * np.pi * x) + C

x_data = df["x"]
y_data = df["y"]

popt, pcov = curve_fit(func, x_data, y_data)

A = popt[0]
B = popt[1]
C = popt[2]
print(f"係数 A: {A}, B: {B}, C: {C}")

plt.scatter(x_data, y_data, label="観測データ", color="blue")
plt.plot(x_data, func(x_data, A, B, C), label="予測値", color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
