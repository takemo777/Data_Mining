from sklearn import random_projection
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

X_TSNEprojected = TSNE(n_components=2, random_state=0).fit_transform(digits.data)

plt.scatter(X_TSNEprojected[:,0], X_TSNEprojected[:,1], c=digits.target,alpha=0.5, cmap='rainbow')
plt.colorbar()
plt.show()
