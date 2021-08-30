from data import get_data_between_dates, trim_data
from eval import eval_models
from plots import plot_predictions, plot_bike_availability_graphs, plot_linear_cross_validation, \
    plot_linear_feature_engineering, plot_knn_cross_validation, plot_knn_cross_validation_2, \
    plot_knn_feature_engineering


if __name__ == '__main__':
    trim_data(["65", "34"])
    t, y = get_data_between_dates(34, "03−02−2020", "29−03−2020")
    # plot_bike_availability_graphs()
    # plot_knn_feature_engineering(t, y, feedforward=False, specify_weekends=0)
    # plot_knn_cross_validation(t, y); plot_knn_cross_validation_2(t, y)
    # plot_linear_feature_engineering(t, y, feedforward=False, specify_weekends=0)
    # plot_linear_cross_validation(t, y)
    # plot_predictions(30, 31, 34, 12)
    # eval_models(12, 65)
