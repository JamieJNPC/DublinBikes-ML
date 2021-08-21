import numpy as np
import math
import matplotlib.pyplot as plt
import constants
from data import get_data_between_dates, trim_data, get_feature_engineered_data
from knn import knn_cross_validate, knn_predict_q_future_nodes
from plots import plot_bike_availability_graphs, plot_knn_graphs, plot_knn_predictions, plot_knn_feature_engineering, \
    plot_knn_cross_validation


def test_preds(q,dd,lag,plot):
    #q−step ahead prediction
    stride=1
    XX=y[0:y.size-q-lag*dd:stride]
    for i in range(1,lag):
        X=y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX=np.column_stack((XX,X))
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(np.arange(0,yy.size), test_size=0.2)
    from sklearn.linear_model import Ridge
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t, y, color='black'); plt.scatter(tt, y_pred, color='blue')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data", "predictions"],loc='upper right')
        day=math.floor(24*60*60/constants.TIME_BETWEEN_SAMPLES) # number of samples per day
        plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))
        plt.show()


def legacy_code():
    # prediction using short−term trend
    plot = True
    test_preds(q=10, dd=1, lag=3, plot=plot)
    # prediction using daily seasonality
    d = math.floor(24 * 60 * 60 / constants.TIME_BETWEEN_SAMPLES)  # number of samples per day
    test_preds(q=d, dd=d, lag=3, plot=plot)
    # prediction using weekly seasonality
    w = math.floor(d * 7)  # number of samples per day
    test_preds(q=w, dd=w, lag=3, plot=plot)

    # putting it together
    q = 10
    lag = 3;
    stride = 1
    len = y.size - w - lag * w - q
    XX = y[q:q + len:stride]
    for i in range(1, lag):
        X = y[i * w + q:i * w + q + len:stride]
        XX = np.column_stack((XX, X))
    for i in range(0, lag):
        X = y[i * d + q:i * d + q + len:stride]
        XX = np.column_stack((XX, X))
    for i in range(0, lag):
        X = y[i:i + len:stride]
        XX = np.column_stack((XX, X))
    yy = y[lag * w + w + q:lag * w + w + q + len:stride]
    tt = t[lag * w + w + q:lag * w + w + q + len:stride]
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    # train = np.arange(0,yy.size)
    from sklearn.linear_model import Ridge
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t, y, color='black');
        plt.scatter(tt, y_pred, color='blue')
        plt.xlabel("time (days)");
        plt.ylabel("#bikes")
        plt.legend(["training data", "predictions"], loc='upper right')
        plt.xlim((4 * 7, 4 * 7 + 4))
        plt.show()

# Stations we are interested in are 34 and 65
# Mar 15 for normal times, Mar 29 for end times
if __name__ == '__main__':
    trim_data(["65", "34"])
    # plot_bike_availability_graphs()
    t, y = get_data_between_dates(34, "03−02−2020", "29−03−2020")
    t2, y2 = get_feature_engineered_data(t, y, 1, 1, 2, 0)
    # plot_knn_predictions(t2, y2)
    # plot_knn_feature_engineering(t, y)
    plot_knn_cross_validation(t, y)
    # knn_cross_validate(t, y, [1, 5, 20, 50])
    # legacy_code()
