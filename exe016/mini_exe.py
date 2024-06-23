from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics import r2_score

df = pd.read_csv("reg_data.csv")

test_X = df["x"].to_numpy().copy().reshape(-1, 1)
test_t = df["t"].to_numpy().copy()

model = LinearRegression()

model.fit(test_X, test_t)

pred_y = model.predict(test_X)

r2 = r2_score(test_t, pred_y)

print("決定係数R2=",end="")
print(r2_score(test_t,pred_y))

plt.scatter(df["x"],df["t"])
plt.xlabel("x")
plt.ylabel("t")
plt.scatter(df["x"],df["t"],label="観測値")
plt.title("x-t")
plt.plot(test_X,pred_y,color="red",label="予測値")
plt.legend()
plt.savefig("mini016_exe_01.png")
plt.show()
