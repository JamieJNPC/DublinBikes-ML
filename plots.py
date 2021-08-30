import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import constants
import knn
import linear
from data import get_data_between_dates, get_feature_engineered_data, predict_q_nodes_ahead
from eval import make_dummy_predictions
from knn import knn_predict_q_future_nodes, gaussian_kernel_0, gaussian_kernel_10, gaussian_kernel_25,\
    gaussian_kernel_1, gaussian_kernel_5


def plot_bike_availability(x, y, xlabel, ylabel, xticks, xticklabels, ax):
    ax.scatter(x, y, color='red', marker='.', s=4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_bike_availability_graphs():
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    plt.rcParams['figure.dpi'] = 600
    # Plot entire range
    t, y = get_data_between_dates(34, "01−01−2020", "01−04−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 31, 60, 91],
                           ["Jan 1", "Feb 1", "Mar 1", "Apr 1"], axs[0, 0])

    # Plot range to use in program
    t, y = get_data_between_dates(34, "03−02−2020", "29−03−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 10, 20, 30, 40, 50],
                           ["Feb 3", "Feb 13", "Feb 23", "Mar 4", "Mar 14", "Mar 24"], axs[2, 0])

    # Plot data for one week
    t, y = get_data_between_dates(34, "03−02−2020", "10−02−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 1, 2, 3, 4, 5, 6],
                           ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"], axs[1, 0])
    # Plot entire range
    t, y = get_data_between_dates(65, "01−01−2020", "01−04−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 31, 60, 91],
                           ["Jan 1", "Feb 1", "Mar 1", "Apr 1"], axs[0, 1])

    # Plot range to use in program
    t, y = get_data_between_dates(65, "03−02−2020", "29−03−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 10, 20, 30, 40, 50],
                           ["Feb 3", "Feb 13", "Feb 23", "Mar 4", "Mar 14", "Mar 24"], axs[2, 1])

    # Plot data for one week
    t, y = get_data_between_dates(65, "03−02−2020", "10−02−2020")
    plot_bike_availability(t, y, "Date", "# of Bikes Available", [0, 1, 2, 3, 4, 5, 6],
                           ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"], axs[1, 1])

    axs[0, 0].annotate("Portobello Harbour", xy=(0.5, 1.05), xycoords='axes fraction', size=24, ha='center', va='baseline')
    axs[0, 1].annotate("Convention Centre", xy=(0.5, 1.05), xycoords='axes fraction', size=24, ha='center', va='baseline')
    axs[0, 0].annotate("Entire Range", xy=(-0.125, 0.5), xycoords='axes fraction', size=24, rotation=90, ha='right', va='center')
    axs[1, 0].annotate("One Week", xy=(-0.125, 0.5), xycoords='axes fraction', size=24, rotation=90, ha='right', va='center')
    axs[2, 0].annotate("Usable Data", xy=(-0.125, 0.5), xycoords='axes fraction', size=24, rotation=90, ha='right', va='center')
    fig.tight_layout()
    fig.show()
    plt.rcParams['figure.dpi'] = 100


def generate_plot_with_training_data(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], y, color="red", label="training data")
    return fig, ax


def plot_prediction_data(X, y, ax, color, label):
    ax.plot(X[:,0], y, color=color, label=label)


def plot_feature_prediction_error(xaxis, mean, error, x, ax):
    if x == 0:
        ax.errorbar(xaxis[1:], mean[1:], error[1:], label="0 previous weeks")
    else:
        ax.errorbar(xaxis, mean, error, label="{} previous weeks".format(x))


def plot_knn_feature_engineering(t, y, feedforward=False, specify_weekends=0):
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axvspan(-0.5, 3.5, color='red', alpha=0.25)
    ax.axvspan(3.5, 7.5, color='green', alpha=0.25)
    ax.axvspan(7.5, 11.5, color='yellow', alpha=0.25)
    ax.axvspan(11.5, 15.5, color='purple', alpha=0.25)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xticklabels([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    ax.set_yscale('log')
    ax.set_ylim(0, 40)
    ax.set_yticks([1, 5, 10, 15, 20, 30, 40])
    ax.set_yticklabels([1, 5, 10, 15, 20, 30, 40])
    for x in [0,1,2,3]:
        print("Generating error bar for knn with %i week's backtracking" % x)
        # Nodes to go back and days to go back
        backsizes = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3],
                     [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
        for i in backsizes:
            i.append(x)
        backsize_mpse = [knn.knn_feature_engineering(t, y, 2, backsize, specify_weekends=specify_weekends, feedforward=feedforward) for backsize in backsizes]
        backsize_mean = [np.mean(x) for x in backsize_mpse]
        backsize_std = [np.std(x) for x in backsize_mpse]
        plot_feature_prediction_error(range(len(backsizes)), backsize_mean, backsize_std, x, ax)
    ax.set_xlabel("Number of previous nodes in features")
    ax.set_ylabel("Mean Squared Prediction Error (log)")
    fig.legend()
    fig.show()


def plot_knn_cross_validation(t, y):
    plt.rcParams['figure.dpi'] = 300
    ks = [1, 5, 10, 20, 50, 100]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in ["uniform", "distance", gaussian_kernel_0, gaussian_kernel_1, gaussian_kernel_5,
                   gaussian_kernel_10, gaussian_kernel_25]:
        print(weight)
        k_mpse = [knn.knn_cross_validate(t, y, k, weight) for k in ks]
        k_mean = [np.mean(x) for x in k_mpse]
        k_std = [np.std(x) for x in k_mpse]
        ax.errorbar(ks, k_mean, k_std)
    ax.set_xticks(ks)
    ax.set_yscale("log")
    ax.set_yticks([1, 5, 10, 50, 100, 500])
    ax.set_yticklabels([1, 5, 10, 50, 100, 500])
    ax.set_ylim(1, 500)
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Squared Prediction Error (Log)")
    fig.legend(["uniform", "distance", "gaussian with sigma=0", "gaussian with sigma=1", "gaussian with sigma=5",
                "gaussian with sigma=10", "gaussian with sigma=25"])
    fig.show()


def plot_knn_cross_validation_2(t, y):
    plt.rcParams['figure.dpi'] = 300
    ks = [1, 5, 10, 20, 50, 100]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in ["uniform", "distance", gaussian_kernel_0]:
        print(weight)
        k_mpse = [knn.knn_cross_validate(t, y, k, weight) for k in ks]
        k_mean = [np.mean(x) for x in k_mpse]
        k_std = [np.std(x) for x in k_mpse]
        ax.errorbar(ks, k_mean, k_std)
    ax.set_xticks(ks)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Squared Prediction Error")
    fig.legend(["uniform", "distance", "gaussian with sigma=0"])
    fig.show()


def plot_linear_feature_engineering(t, y, feedforward=False, specify_weekends=0):
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axvspan(-0.5, 3.5, color='red', alpha=0.25)
    ax.axvspan(3.5, 7.5, color='green', alpha=0.25)
    ax.axvspan(7.5, 11.5, color='yellow', alpha=0.25)
    ax.axvspan(11.5, 15.5, color='purple', alpha=0.25)
    ax.set_xlim(-0.5, 15.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xticklabels([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    #ax.set_yticks([0, 1, 2, 3, 4, 5])
    #ax.set_yticklabels([0, 1, 2, 3, 4, 5])
    #ax.set_ylim(0, 5)
    for x in [0, 1, 2, 3]:
        print("Generating error bar for linear with %i week's backtracking" % x)
        # Nodes to go back and days to go back
        backsizes = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3],
                     [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
        for i in backsizes:
            i.append(x)
        backsize_mpse = [linear.linear_feature_engineering(t, y, 2, backsize, specify_weekends=specify_weekends, feedforward=feedforward) for backsize in backsizes]
        backsize_mean = [np.mean(x) for x in backsize_mpse]
        backsize_std = [np.std(x) for x in backsize_mpse]
        plot_feature_prediction_error(range(len(backsizes)), backsize_mean, backsize_std, x, ax)
    ax.set_xlabel("Number of previous nodes in features")
    ax.set_ylabel("Mean Squared Prediction Error")
    fig.legend()
    fig.show()

def plot_linear_cross_validation(t, y):
    cs = [0.1, 0.5, 1, 10, 100, 1000]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for regularisation in ["l1", "l2"]:
        c_mpse = [linear.linear_cross_validate(t, y, c, regularisation) for c in cs]
        c_mean = [np.mean(x) for x in c_mpse]
        c_std = [np.std(x) for x in c_mpse]
        ax.errorbar(cs, c_mean, c_std)
    ax.set_xscale("log")
    ax.set_xticks(cs)
    ax.set_xticklabels(["0.1", "0.5", "1", "10", "100", "1000"])
    ax.set_xlabel("Penalty Weight C (log)")
    ax.set_ylabel("Mean Squared Prediction Error")
    fig.legend(["L1 (Lasso)", "L2 (Ridge)"])
    fig.show()


def plot_predictions(start_time, end_time, station, q):
    t, y = get_data_between_dates(station, "03−02−2020", "29−03−2020")
    t2, y2 = get_feature_engineered_data(t, y, 1, 0, 3, specify_weekends=0)
    linear_model = Ridge(alpha=1/2)
    if station == 34:
        t3, y3 = get_feature_engineered_data(t, y, 1, 1, 1, specify_weekends=0)
        knn_model = KNeighborsRegressor(n_neighbors=50, weights="distance")
    else:
        t3, y3 = get_feature_engineered_data(t, y, 1, 3, 1, specify_weekends=0)
        knn_model = KNeighborsRegressor(n_neighbors=10, weights="distance")

    x = np.column_stack((t2, y2))
    xtrain = []; xtest = []
    for a in x:
        if a[0] < start_time or a[0] > end_time:
            xtrain.append(a.tolist())
        else:
            xtest.append(a.tolist())
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)

    ttrain = xtrain[:, 0:-1]; ytrain = xtrain[:, -1]
    ttest = xtest[:, 0:-1]; ytest = xtest[:, -1]

    x = np.column_stack((t3, y3))
    x3train = [];
    x3test = []
    for a in x:
        if a[0] < start_time or a[0] > end_time:
            x3train.append(a.tolist())
        else:
            x3test.append(a.tolist())
    x3train = np.array(x3train)
    x3test = np.array(x3test)
    t3train = x3train[:, 0:-1]; y3train = x3train[:, -1]
    t3test = x3test[:, 0:-1]; y3test = x3test[:, -1]

    xpred, yreal, ypred = predict_q_nodes_ahead(ttest, ytest, q, ttrain, ytrain, linear_model)
    xpred2, yreal2, ypred2 = predict_q_nodes_ahead(t3test, y3test, q, t3train, y3train, knn_model)

    fig, ax = plt.subplots()
    ax.scatter(t, y, color="red", label="Real Data")
    # ax.plot(xpred[:, 0], ypred, color="blue", label="Ridge Prediction")
    # ax.plot(xpred2[:, 0], ypred2, color="green", label="kNN Prediction")
    ax.set_xlabel("Days from beginning of dataset")
    ax.set_ylabel("# of Bikes Available")
    xpred, ypred = make_dummy_predictions(ttest[:, 0], ytest, q)
    ax.plot(xpred, ypred, color="blue", label="Baseline Prediction")
    ax.set_xlim(start_time, end_time)
    fig.legend()
    fig.show()

