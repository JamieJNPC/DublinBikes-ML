import math

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from data import get_feature_engineered_data, predict_q_nodes_ahead


def linear_feature_engineering(x, y, q, backsizes, specify_weekends, feedforward=False):
    x, y = get_feature_engineered_data(x, y, backsizes[0], backsizes[1], backsizes[2], specify_weekends=specify_weekends)
    kf = KFold(n_splits=5)
    results = []
    for train, test in kf.split(x):
        c = 1; model = Ridge(alpha=1/(2*c))
        if feedforward:
            xtrain = x[train][:-q]; ytrain = y[train][q:]; xtest = x[test][:-q]; ytest = y[test][q:]
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            yreal = ytest
        else:
            xpred, yreal, ypred = predict_q_nodes_ahead(x[test], y[test], q, x[train], y[train], model)
        results.append(mean_squared_error(yreal, ypred))
    return results


def linear_cross_validate(x, y, c, regularisation):
    q = 2
    x, y = get_feature_engineered_data(x, y, 1, 1, 2, specify_weekends=0)
    kf = KFold(n_splits=5)
    results = []
    for train, test in kf.split(x):
        if regularisation == "l2":
            model = Ridge(alpha=1/(2*c))
        else:
            model = Lasso(alpha=1/(2*c))
        xpred, yreal, ypred = predict_q_nodes_ahead(x[test], y[test], q, x[train], y[train], model)
        results.append(mean_squared_error(yreal, ypred))
    return results