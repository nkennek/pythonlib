#!/usr/bin/env python
# -- coding:utf-8 --

import numpy as np
import itertools
import pandas as pd

class RobustLinearRegressor(object):
    '''
    多項式f(x)による線形回帰のhuber回帰及びtukey回帰
    '''
    def __init__(self, x, y, dim = 2, eta = 1.0, how = "huber", randomize_leverage = 1.0, iter_num = 10):
        self.x, self.y = x, y #標本
        self.dim = dim #多項式の最大次数
        self.eta = eta #損失関数の振る舞いを決定する係数．大きいほど二乗誤差に近く
        if how not in ["huber", "tukey"]:
            raise ValueError("algorithm of loss function must be either 'huber' or 'tukey'")
        self.how = how
        self.iter_num = iter_num
        self.phi = self._design_matrix()
        self._initialize_weights(leverage = randomize_leverage)

    def _initialize_weights(self, leverage = 1.0):
        self.weights = np.random.random((self.dim + 1,)) * leverage
        return self

    def _design_matrix(self):
        n = self.x.size if self.dim == 1 else self.x.shape[0]
        phi = np.ones((n, self.dim + 1))
        if self.dim == 1:
            phi[:, 0] = self.x
        else:
            for i in range(self.dim):
                phi[:,i] = self.x[:, i]
        return phi

    def _sample_weight(self):
        how = self.how
        residuals = np.abs(self.predict(self.x) - self.y)
        lower_mask = (residuals <= self.eta)
        upper_mask = np.invert(lower_mask)
        if how == "huber":
            residuals[lower_mask] = 1.0
            residuals[upper_mask] = self.eta / residuals[upper_mask]
        else: # if how == "tukey":
            residuals[lower_mask] = (1.0 - residuals[lower_mask]**2 / self.eta**2)**2
            residuals[upper_mask] = 0
        weights_diag = np.zeros((len(residuals), len(residuals)))
        for i in range(len(residuals)):
            weights_diag[i][i] = residuals[i]
        return weights_diag


    def _update(self):
        how = self.how
        weights = self.weights
        sample_weights = self._sample_weight()
        #if how == "huber":
        #    #self.weights = np.dot(np.linalg.inv(sample_weights), self.y)
        #    tmp_matrix_for_calc = np.dot(np.transpose(self.phi), sample_weights)
        #    term1 = np.linalg.inv(np.dot(tmp_matrix_for_calc, self.phi))
        #    term2 = np.dot(tmp_matrix_for_calc, self.y)
        #    self.weights = np.dot(term1, term2)
        #else: # if how == "tukey":
        #    ##TODO tukeyの実装
        #    pass
        tmp_matrix_for_calc = np.dot(np.transpose(self.phi), sample_weights)
        term1 = np.linalg.inv(np.dot(tmp_matrix_for_calc, self.phi))
        term2 = np.dot(tmp_matrix_for_calc, self.y)
        self.weights = np.dot(term1, term2)
        return self

    def fit(self):
        for _ in range(self.iter_num):
            self._update()
        return self

    def predict(self, X):
        return np.sum(X[:, np.newaxis] * self.weights[1:], axis = 1) + self.weights[0]

    #パラメータ更新１回ごとに予測して成績を測る
    def test(self, X_test, y_test):
        score_table = pd.DataFrame(columns = ["iter_times", "x", "true", "predict"])

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
        if "eta" in keys:
            self.eta = params["eta"]
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
    eta_values = np.arange(0.1, 1, 0.1)

    score_table = pd.DataFrame(columns = ["h", "eta", "mean(mse)"])

    #generate samples
    np.random.seed(0)
    X = np.arange(0, 10, 0.05)
    #X = X.reshape((1, len(X)))
    y = np.sin(X) + 0.1 * X  + np.array([0.2*np.random.rand() for _ in range(len(X))])

    for h, eta in itertools.product(h_values, eta_values):
        logger.debug("processing with h: {h}, eta: {eta}".format(h = h, eta = eta))
        cv = CrossValidation(RobustLinearRegressor, X, y, clf_params = {"h":h, "eta":eta})
        scores = cv.score()
        score = np.array(scores).mean()
        if score > 1.0e5:
            logger.debug("warning! score is unreasonably high: {score}".format(score = score))
        record = pd.Series([h, eta, score], index = ["h", "eta", "mean(mse)"])
        score_table = score_table.append(record, ignore_index = True)
    logger.debug("output csv file...")
    score_table.to_csv("cv_result_robust.csv")
    logger.debug("done")
