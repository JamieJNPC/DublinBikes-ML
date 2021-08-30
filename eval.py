import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data import get_feature_engineered_data, predict_q_nodes_ahead, get_data_between_dates


def make_dummy_predictions(x, y, q):
    return x[q:], y[:-q]


def eval_models(q, station):
    kf = KFold(n_splits=5)
    results = [[], [], [], [], [], []]
    t, y = get_data_between_dates(station, "03−02−2020", "29−03−2020")
    t2, y2 = get_feature_engineered_data(t, y, 1, 0, 3, specify_weekends=0)
    linear_model = Ridge(alpha=1/2)
    if station == 34:
        t3, y3 = get_feature_engineered_data(t, y, 1, 1, 1, specify_weekends=0)
        knn_model = KNeighborsRegressor(n_neighbors=50, weights="distance")
    else:
        t3, y3 = get_feature_engineered_data(t, y, 1, 3, 1, specify_weekends=0)
        knn_model = KNeighborsRegressor(n_neighbors=10, weights="distance")

    print("Evaluating Station %i" % station)
    for train, test in kf.split(t2):
        xpred, yreal, ypred = predict_q_nodes_ahead(t2[test], y2[test], q, t2[train], y2[train], linear_model)
        results[0].append(r2_score(yreal, ypred)); results[1].append(mean_squared_error(yreal, ypred)); results[2].append(mean_absolute_error(yreal, ypred))
    for train, test in kf.split(t3):
        xpred2, yreal2, ypred2 = predict_q_nodes_ahead(t3[test], y3[test], q, t3[train], y3[train], knn_model)
        results[3].append(r2_score(yreal2, ypred2)); results[4].append(mean_squared_error(yreal2, ypred2)); results[5].append(mean_absolute_error(yreal2, ypred2))

    print("Linear mean R2 Score: ", np.mean(results[0])); print("Linear mean MSPE Score: ", np.mean(results[1])); print("Linear mean MAE Score: ", np.mean(results[2]))
    print("Linear coefs: ", linear_model.coef_)
    print("kNN mean R2 Score: ", np.mean(results[3])); print("kNN mean MSPE Score: ", np.mean(results[4])); print("kNN mean MAE Score: ", np.mean(results[5]))

    xpred, ypred = make_dummy_predictions(t2, y2, q)
    print("Dummy R2 Score: ", r2_score(y2[q:], ypred)); print("Dummy MSPE Score: ", mean_squared_error(y2[q:], ypred)); print("Dummy MAE Score: ", mean_absolute_error(y2[q:], ypred))
