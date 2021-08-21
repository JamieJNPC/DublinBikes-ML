import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import constants
import knn
from data import get_data_between_dates
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

    axs[0, 0].annotate("Portobello", xy=(0.5, 1.05), xycoords='axes fraction', size=24, ha='center', va='baseline')
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


def plot_knn_graphs(t, y):
    plt.rcParams['figure.dpi'] = 600
    X = t[np.where(t[:,0] < 26)]
    yy = y[:len(X)]
    trainX, testX, trainY, testY = train_test_split(X, yy, test_size=0.2)
    # X = X.reshape(-1, 1)
    fig, ax = generate_plot_with_training_data(X, yy)
    testX = testX[testX[:,0].argsort()]
    plot_prediction_data(X=testX, y=(KNeighborsRegressor(n_neighbors=1).fit(trainX, trainY).predict(testX)), ax=ax,
                         label="k = 1", color="blue")
    plot_prediction_data(X=testX, y=(KNeighborsRegressor(n_neighbors=2).fit(trainX, trainY).predict(testX)), ax=ax,
                         label="k = 2", color="green")
    plot_prediction_data(X=testX, y=(KNeighborsRegressor(n_neighbors=5).fit(trainX, trainY).predict(testX)), ax=ax,
                         label="k = 5", color="purple")
    plot_prediction_data(X=testX, y=(KNeighborsRegressor(n_neighbors=10).fit(trainX, trainY).predict(testX)), ax=ax,
                         label="k = 10", color="yellow")

    ax.set_xlabel = "x"
    ax.set_ylabel = "y"
    ax.legend()
    fig.show()
    plt.rcParams['figure.dpi'] = 100


def plot_knn_predictions(t, y):
    plt.rcParams['figure.dpi'] = 300
    trainX = []; trainY = []; testX = []; testY = [];
    for i in range(len(t)):
        if constants.TESTING_START_DAY < t[:, 0][i] < constants.TESTING_END_DAY:
            testX.append(t[i])
            testY.append(y[i])
        else:
            trainX.append(t[i])
            trainY.append(y[i])
    testX = np.asarray(testX)
    ypred = knn_predict_q_future_nodes(trainX, trainY, testX, constants.Q)
    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(testX[:,0], testY, color="red", label="real data")
    ax.plot(testX[:,0][:-constants.Q], ypred, color="blue", label="prediction from 1 hour before")
    ax.set_xlabel = "x"
    ax.set_ylabel = "y"
    ax.legend()
    fig.show()
    plt.rcParams['figure.dpi'] = 100


def plot_knn_feature_prediction_error(xaxis, mean, error):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(xaxis, mean, error)
    ax.set_xticks(xaxis)
    ax.axvspan(-0.5, 3.5, color='red', alpha=0.25)
    ax.axvspan(3.5, 7.5, color='green', alpha=0.25)
    ax.axvspan(7.5, 11.5, color='yellow', alpha=0.25)
    ax.axvspan(11.5, 15.5, color='purple', alpha=0.25)
    ax.set_xticklabels([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
    ax.set_yticks([4,5,6,7,8,9,10,11,12,13,14])
    ax.set_yticklabels([4,"",6,"",8,"",10,"",12,"",14])
    red_patch = mpatches.Patch(color='red', label='1 Previous day', alpha=0.25)
    green_patch = mpatches.Patch(color='green', label='2 Previous days', alpha=0.25)
    yellow_patch = mpatches.Patch(color='yellow', label='3 Previous days', alpha=0.25)
    purple_patch = mpatches.Patch(color='purple', label='4 Previous days', alpha=0.25)
    fig.legend(handles=[red_patch, green_patch, yellow_patch, purple_patch])
    return ax, fig


def plot_knn_feature_engineering(t, y):
    plt.rcParams['figure.dpi'] = 300
    for x in [0,1,2,3]:
        print("Generating error plot for knn with %i week's backtracking" % x)
        # Nodes to go back and days to go back
        backsizes = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4],
                     [3, 1], [3, 2], [3, 3], [3, 4], [4, 1], [4, 2], [4, 3], [4, 4]]
        for i in backsizes:
            i.append(x)
        backsize_mpse = [knn.knn_feature_engineering(t, y, 2, backsize, specify_weekends=0) for backsize in backsizes]
        backsize_mean = [np.mean(x) for x in backsize_mpse]
        backsize_std = [np.std(x) for x in backsize_mpse]
        ax, fig = plot_knn_feature_prediction_error(range(len(backsizes)), backsize_mean, backsize_std)
        ax.set_xlabel("Number of previous nodes in features")
        ax.set_ylabel("Mean Squared Prediction Error")
        ax.set_title("%i Previous Weeks Back" % x)
        fig.show()


def plot_knn_cross_validation(t, y):
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
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Squared Prediction Error (Log)")
    fig.legend(["uniform", "distance", "gaussian with sigma=0", "gaussian with sigma=1", "gaussian with sigma=5",
                "gaussian with sigma=10", "gaussian with sigma=25"])
    fig.show()