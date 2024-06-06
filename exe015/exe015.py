import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import sympy as sp
from sklearn.linear_model import LinearRegression

df = pd.read_csv("reg_data.csv")
plt.scatter(df["x"],df["t"])
plt.xlabel("x")
plt.ylabel("t")
plt.scatter(df["x"],df["t"])
plt.title("x-t")
plt.savefig("x_t.png")
plt.show()

x = df["x"].to_numpy() #必要に応じてdf["x"].to_numpy().reshape(-1,1)
t = df["t"].to_numpy() #必要に応じてdf["t"].to_numpy().reshape(-1,1)

def sol03(x, t):
    
    # E = 0
    a, b = sp.symbols('a b')
    E = np.sum((t - (a * x + b)) ** 2)
    
    """for i in range(0, len(x)):
        x_i = x[i][0]
        E += (t[i] - (a * x_i + b)) ** 2"""

    # 偏微分
    E_diff_a = sp.diff(E, a)
    E_diff_b = sp.diff(E, b)

    # 連立方程式を解く
    result = sp.solve([E_diff_a, E_diff_b], [a, b])
    
    return [result[a], result[b]]

def sol06(x, t):
    
    model = LinearRegression()
    x = x.reshape(-1, 1)
    model.fit(x, t)
    a = model.coef_[0]
    b = model.intercept_
    
    return [a, b]

a3, b3 = sol03(x, t)
a6, b6 = sol06(x, t)

print(f"③の方法で解いたa={a3}, b={b3}")
print(f"⑥の方法で解いたa={a6}, b={b6}")
