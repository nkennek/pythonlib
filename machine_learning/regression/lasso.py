#!/usr/bin/env python
# -- coding:utf-8 --

import numpy as np
import itertools
import pandas as pd

class GaussianKernelLinearRegressor(object):
    #初期化
    def __init__(self, x, y, h = 0.1, l1 = 0.1, iter_num = 100):
        self.x, self.y = x, y #標本
        self.h = h #カーネルバンド幅
        self.l1 = l1 # l1正則化係数
        self.iter_num = iter_num #更新回数
        self._initialize_weights()

    def _gaussian_kernel(self, x1, x2, h):
        return np.exp(-(x1 - x2) ** 2 / (2 * (h ** 2)))

    def _triang_polynomial(self, x):
        #三角多項式で基底を計算
        max_number = 15
        arr = np.array([[1] * x.size])
        for i in range(1, max_number):
            arr = np.append(arr, [np.sin(i*x)/2], axis = 0)
            arr = np.append(arr, [np.cos(i*x)/2], axis = 0)
        return arr.transpose()

    def _design_matrix(self):
        n = self.x.size
        phi = np.zeros((n,n))
        for i, j in itertools.product(range(n), range(n)):
            phi[i][j] = self._gaussian_kernel(self.x[i], self.x[j], self.h)
        self.phi = phi
        return phi

    def _initialize_weights(self):
        self.weights = np.random.random((self.x.size,))
        self.u = np.random.random((self.x.size,))
        self.z = self.weights
        return self

    def _update(self):
        weights = self.weights
        u = self.u
        phi = self.phi
        z = self.z
        y = self.y
        #重みの更新
        coef1 = np.linalg.inv(np.dot(phi.transpose(), phi) + np.eye(phi.shape[0]))
        coef2 = np.dot(phi.transpose(), y) + z - u
        weights = np.dot(coef1, coef2)
        #zの更新
        z = weights + u
        z[np.abs(z) < self.l1] = 0
        #uの更新
        u = u + weights - z

        self.weights = weights
        self.z = z
        self.u = u
        return self

    def fit(self):
        phi = self._design_matrix()
        for _ in range(self.iter_num):
            self._update()
        return self

    def predict(self, X):
        predict_val = 0
        for i in range(self.x.size):
            predict_val += self.weights[i] * self._gaussian_kernel(X, self.x[i], self.h)
        return predict_val

    #パラメータ更新１回ごとに予測して成績を測る
    def test(self, X_test, y_test):
        score_table = pd.DataFrame(columns = ["iter_times", "x", "true", "predict"])
        phi = self._design_matrix()

        for i in range(1, self.iter_num + 1):
            self._update()
            pred_y = self.predict(X_test)
            ith    = pd.DataFrame(zip([i]*len(pred_y), X_test.tolist(), y_test.tolist(), pred_y.tolist()), columns = score_table.columns)
            score_table = score_table.append(ith, ignore_index = True)

        return pd.melt(score_table, id_vars = ["iter_times", "x"], value_vars= ["true", "predict"])

    def update_params(self, params):
        keys = params.keys()
        if "h" in keys:
            self.h = params["h"]
        if "l1" in keys:
            self.l1 = params["l1"]
        return self

if __name__ == "__main__":
    from logging import getLogger, StreamHandler, DEBUG
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    #デバッガ
    #import pdb; pdb.set_trace()
    #交差検定によって制度検証
    import sys
    import os
    sys.path.append(os.path.join(os.environ['HOME'], 'Projects', 'pythonlib', 'machine_learning', 'validation'))
    from cross_validation import CrossValidation
    import pandas as pd

    h_values = np.arange(0.1, 1, 0.1)
    l1_values = np.arange(0.1, 1, 0.1)

    score_table = pd.DataFrame(columns = ["h", "l1", "mean(mse)"])

    #generate samples
    np.random.seed(0)
    X = np.arange(0, 10, 0.05)
    y = np.sin(X) + 0.1 * X  + np.array([0.2*np.random.rand() for _ in range(len(X))])

    for h, l1 in itertools.product(h_values, l1_values):
        logger.debug("processing with h: {h}, l1: {l1}".format(h = h, l1 = l1))
        cv = CrossValidation(GaussianKernelLinearRegressor, X, y, clf_params = {"h":h, "l1":l1})
        scores = cv.score()
        score = np.array(scores).mean()
        if score > 1.0e5:
            logger.debug("warning! score is unreasonably high: {score}".format(score = score))
        record = pd.Series([h, l1, score], index = ["h", "l1", "mean(mse)"])
        score_table = score_table.append(record, ignore_index = True)
    logger.debug("output csv file...")
    score_table.to_csv("cv_result_lasso.csv")
    logger.debug("done")
