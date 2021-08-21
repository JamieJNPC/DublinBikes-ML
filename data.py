import csv
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import constants


def trim_data(station_ids):
    with open('dublinbikes_20200101_20200401.csv', 'r') as sourcefile:
        outputfile = open('data.csv', 'w')
        sourcereader = csv.reader(sourcefile)
        outputwriter = csv.writer(outputfile)
        for i, row in enumerate(sourcereader):
            if i == 0 or row[0] in station_ids:
                outputwriter.writerow(row)


def get_data_between_dates(station_id, start_date, end_date):
    # read data. column 1 is date/time, col 6 is #bikes
    df = pd.read_csv("data.csv", usecols=[0, 1, 6], parse_dates=[1])
    df = df[df.iloc[:, 0] == station_id]
    # 3rd Feb 2020 is a monday, 10th is following monday
    start = pd.to_datetime(start_date, format='%d−%m−%Y')
    end = pd.to_datetime(end_date, format='%d−%m−%Y')
    # convert date/time to unix timestamp in sec and calc interval between samples
    t_full = pd.array(pd.DatetimeIndex(df.iloc[:, 1]).view(np.int64)) / 1000000000
    # extract data between start and end dates
    t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
    t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
    t = np.extract([(t_full >= t_start[0]).to_numpy() & (t_full <= t_end[0]).to_numpy()], t_full)
    t = (t - t[0]) / 60 / 60 / 24  # convert timestamp to days
    y = np.extract([(t_full >= t_start[0]).to_numpy() & (t_full <= t_end[0]).to_numpy()], df.iloc[:, 2]).astype(np.int64)
    return t, y


def get_feature_engineered_data(x, y, num_prev_points, num_prev_days, num_prev_weeks, specify_weekends=0):
    shift = constants.SAMPLES_BETWEEN_WEEKS*num_prev_weeks
    x1 = x[shift:]
    y1 = y[shift:]
    x2 = []
    for i, xval in enumerate(x1):
        temp = [xval]
        for j in range(num_prev_points):
            temp.append(y[shift + i - j])
        for j in range(num_prev_days):
            temp.append(y[shift + i - (j * constants.SAMPLES_BETWEEN_DAYS)])
            if specify_weekends == 1:
                temp.append((xval - j) % 7 >= 5)
            elif specify_weekends == 2:
                temp.append(math.floor(xval - j) % 7)
        for j in range(num_prev_weeks):
            temp.append(y[shift + i - (j * constants.SAMPLES_BETWEEN_WEEKS)])
        x2.append(np.asarray(temp))
    return np.asarray(x2), y1

