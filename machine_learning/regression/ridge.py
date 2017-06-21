#!/usr/bin/env python
# -- coding:utf-8 --

import numpy as np
import itertools

class GaussianKernelLinearRegressor(object):
    #初期化
    def __init__(self, x, y, h = 0.1, l2 = 0.1):
        self.x, self.y = x, y #標本
        self.h = h #カーネルバンド幅
        self.l2 = l2 # l2正則化係数

    def _gaussian_kernel(self, x1, x2, h, axis = None):
        if axis is not None:
            return np.sum(np.exp(-(x1 - x2) ** 2 / (2 * (h ** 2))), axis = axis)
        else:
            return np.exp(-(x1 - x2) ** 2 / (2 * (h ** 2)))

    def _design_matrix(self):
        n = self.x.size if self.x.ndim == 1 else self.x.shape[0]
        #Xが多次元である場合はサンプルごとの距離を一次元に集約するためaxis=1で和をとる
        axis = None if self.x.ndim == 1 else 1
        phi = np.ones((n,n))
        for i in range(n):
            phi[i] = self._gaussian_kernel(self.x, self.x[i], self.h, axis = axis)
        #for i, j in itertools.combinations(range(n), 2):
        #    phi[i][j] = self._gaussian_kernel(self.x[i], self.x[j], self.h)
        #    phi[j][i] = phi[i][j]
        self.phi = phi
        return phi

    def fit(self):
        phi = self._design_matrix()
        eye = np.eye(phi.shape[0])
        term1 = phi.transpose().dot(phi) + self.l2 * eye
        phi_inv = np.linalg.inv(phi)
        weights = np.linalg.inv(term1).dot(phi.transpose()).dot(self.y)
        self.weights = weights

        return self

    def predict(self, X):
        n = self.x.size if self.x.ndim == 1 else self.x.shape[0]
        test_length = X.size if X.ndim == 1 else X.shape[0]
        predict_val = np.zeros((test_length,))
        #Xが多次元である場合はサンプルごとの距離を一次元に集約するためaxis=1で和をとる
        axis = None if self.x.ndim == 1 else 1
        for i in range(n):
            predict_val += self.weights[i] * self._gaussian_kernel(X, self.x[i], self.h, axis = axis)
        return predict_val

    def update_params(self, params):
        keys = params.keys()
        if "h" in keys:
            self.h = params["h"]
        if "l2" in keys:
            self.l2 = params["l2"]
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
    import pandas as pd
    import sys
    import os
    sys.path.append(os.path.join(os.environ['HOME'], 'Projects', 'pythonlib', 'machine_learning', 'validation'))
    from cross_validation import CrossValidation
    import pandas as pd

    h_values = np.arange(0.1, 3.0, 0.1)
    l2_values = np.arange(0, 1, 0.1)

    score_table = pd.DataFrame(columns = ["h", "l2", "mean(mse)"])

    #generate samples
    np.random.seed(0)
    X = np.arange(0, 10, 0.05)
    #y = np.sin(X) + np.array([0.2*np.random.rand() for _ in range(len(X))])
    y = np.sin(X) + 0.1 * X  + np.array([0.2*np.random.rand() for _ in range(len(X))])

    for h, l2 in itertools.product(h_values, l2_values):
        logger.debug("processing with h: {h}, l2: {l2}".format(h = h, l2 = l2))
        cv = CrossValidation(GaussianKernelLinearRegressor, X, y, clf_params = {"h":h, "l2":l2})
        scores = cv.score()
        score = np.array(scores).mean()
        if score > 1.0e5:
            logger.debug("warning! score is unreasonably high: {score}".format(score = score))
        record = pd.Series([h, l2, score], index = ["h", "l2", "mean(mse)"])
        score_table = score_table.append(record, ignore_index = True)
    logger.debug("output csv file...")
    score_table.to_csv("cv_result_ridge.csv")
    logger.debug("done")
