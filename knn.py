import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

import constants
from data import get_feature_engineered_data, predict_q_nodes_ahead


def gaussian_kernel_0(distances):
    weights = np.exp(-0*(distances**2))
    return weights/np.sum(weights)


def gaussian_kernel_1(distances):
    weights = np.exp(-1*(distances**2))
    return weights/np.sum(weights)


def gaussian_kernel_5(distances):
    weights = np.exp(-5*(distances**2))
    return weights/np.sum(weights)


def gaussian_kernel_10(distances):
    weights = np.exp(-10*(distances**2))
    return weights/np.sum(weights)


def gaussian_kernel_25(distances):
    weights = np.exp(-25*(distances**2))
    return weights/np.sum(weights)


def k_nearest_neighbours(X, y, k, gamma=-1):
    if gamma == 0: output = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel_0)
    elif gamma == 1: output = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel_1)
    elif gamma == 5: output = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel_5)
    elif gamma == 10: output = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel_10)
    elif gamma == 25: output = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel_25)
    else: raise ValueError("Invalid Gamma Value")
    return output

def knn_predict_q_future_nodes(xtrain, ytrain, xtest, q):
    xtrain = xtrain[:-q]; ytrain = ytrain[q:]
    xtest = xtest[:-q]
    model = KNeighborsRegressor(n_neighbors=50).fit(xtrain, ytrain)
    # model = LinearRegression(fit_intercept=True).fit(xtrain, ytrain)
    # print(model.intercept_, model.coef_)
    return model.predict(xtest)


def knn_feature_engineering(x, y, q, backsizes, specify_weekends, feedforward):
    x, y = get_feature_engineered_data(x, y, backsizes[0], backsizes[1], backsizes[2], specify_weekends=specify_weekends)
    kf = KFold(n_splits=5)
    results = []
    for train, test in kf.split(x):
        model = KNeighborsRegressor(n_neighbors=20, weights="uniform")
        if feedforward:
            xtrain = x[train][:-q]; ytrain = y[train][q:]; xtest = x[test][:-q]; ytest = y[test][q:]
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            yreal = ytest
        else:
            xpred, yreal, ypred = predict_q_nodes_ahead(x[test], y[test], q, x[train], y[train], model)
        results.append(mean_squared_error(yreal, ypred))
    return results


def knn_cross_validate(x, y, k, weight):
    q = 2
    x, y = get_feature_engineered_data(x, y, 1, 1, 2, specify_weekends=0)
    kf = KFold(n_splits=5)
    results = []
    for train, test in kf.split(x):
        model = KNeighborsRegressor(n_neighbors=k, weights=weight)
        xpred, yreal, ypred = predict_q_nodes_ahead(x[test], y[test], q, x[train], y[train], model)
        results.append(mean_squared_error(yreal, ypred))
    return results