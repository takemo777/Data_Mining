import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv("ploy_reg.csv")

x = df['x'].values.reshape(-1, 1)
y = df['y'].values

polynomial_features = PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

A = model.coef_[4]
B = model.coef_[3]
C = model.coef_[2]
D = model.coef_[1]
E = model.intercept_

y_pred = model.predict(x_poly)

mse = mean_squared_error(y, y_pred)

plt.scatter(x, y, color='blue', label='実績データ')
plt.plot(x, y_pred, color='red', label='予測')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"MSE: {mse}, A: {A}, B: {B}, C: {C}, D: {D}, E: {E}")
