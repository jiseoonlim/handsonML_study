# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:29:59 2021

@author: jiseon Lim
"""
import numpy as np
from sklearn.datasets import fetch_openml

import matplotlib
import matplotlib.pyplot as plt

# MNIST data download
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int8)

# MNIST 데이터 배열 확인
# X, y = mnist["data"], mnist["target"]
# print(X.shape)
# print(y.shape)

# 그려보기

some_digit = X.to_numpy()[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
            interpolation="nearest")
plt.axis("off")
plt.show()

