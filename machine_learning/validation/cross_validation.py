#!/usr/bin/env python
# -- coding:utf-8 --

import numpy as np
from copy import copy

class CrossValidation(object):
    def __init__(self, clf, samples, ans, cv = 5, random_state = None, clf_params = None):
        np.random.seed(random_state)
        self.clf = clf
        self.clf_params = clf_params
        self.samples = samples
        self.ans = ans
        self.cv = cv #グループ数

    def _shuffle(self):
        p = np.random.permutation(len(self.samples))
        self.samples = self.samples[p]
        self.ans = self.ans[p]
        return self

    def _split_sample(self):
        population_size = self.samples.shape[0]
        cv_size = int(population_size / self.cv)
        surpass = population_size % self.cv
        #サンプルをシャッフル
        self._shuffle()
        self.groups_sample = []
        self.groups_ans = []
        low = 0
        for i in range(self.cv):
            if i != self.cv-1:
                self.groups_sample.append(self.samples[low:low + cv_size])
                self.groups_ans.append(self.ans[low:low + cv_size])
                low += cv_size
            else:
                self.groups_sample.append(self.samples[low:])
                self.groups_ans.append(self.ans[low:])
        return self

    def _data_except(self, i):
        arr = np.array([])
        ans = np.array([])
        for idx in range(self.cv):
            if idx == i:
                continue
            else:
                arr = np.append(arr, self.groups_sample[idx])
                ans = np.append(ans, self.groups_ans[idx])
        return arr, ans

    def _data_within(self, i):
        return self.groups_sample[i], self.groups_ans[i]

    def score(self, how = 'mse'):
        scores = []
        self._split_sample()
        for i in range(self.cv):
            X_train, y_train = self._data_except(i)
            X_test, y_test = self._data_within(i)
            clf = self.clf(X_train, y_train)
            if self.clf_params is not None:
                clf.update_params(self.clf_params)
            clf.fit()
            pred = clf.predict(X_test)
            mse = ((y_test - pred) ** 2).mean()
            scores.append(mse)
        return scores
