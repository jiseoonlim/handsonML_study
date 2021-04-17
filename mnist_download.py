# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:29:59 2021

@author: jiseon Lim
"""
import numpy as np
from sklearn.datasets import fetch_openml #Change mldata -> openml

import matplotlib
import matplotlib.pyplot as plt

# MNIST data download
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int8)

# MNIST 데이터 배열 확인
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

# 그려보기
some_digit = X.to_numpy()[36000] #After 2020 Update openml return pandas Dataframe
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
            interpolation="nearest")
plt.axis("off")
plt.show()

# train set / test set
X_train, X_test, y_train, y_test = X.to_numpy()[:60000], X.to_numpy()[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Training SGDclassifier / 5와 5아님만 구분하는 binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit])) #결과값 FALSE

# 교차 검증
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#about result : 정확도를 분류기의 성능 지표로 선호하지 않음
