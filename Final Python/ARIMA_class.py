import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
%matplotlib inline
import statsmodels.api as sm
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
# from statsmodels.api.tsa.arima_model import ARIMA
from math import sqrt
import yfinance as yf

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class EvaluasiForecasting:
    def rmse_metric(actual, predicted):
        mean_error = np.square(np.subtract(actual,predicted)).mean()
        return math.sqrt(mean_error)

    def mae_metric(actual, predicted):
        y_true, predicted = np.array(actual), np.array(predicted)
        return np.mean(np.abs(actual - predicted))

    def mape_metric(actual, predicted): 
        actual, predicted = np.array(actual), np.array(predicted)
        return np.mean(np.abs((actual - predicted) / actual)) * 100

class DataLoad:
    def read_data(start_date,end_date, symbol_dataset):
        df = []
        df = yf.download(symbol_dataset, start=start_date, end=end_date)
        
        return df

class Preprocessing:
    def splitting_dataset(data):
        train_data, test_data = data[0:int((len(data))*0.8)], data[int((len(data)+1)*0.8):]
        
        return train_data, test_data

class ARIMA_model:
    def forecast_model(train_data, test_data, p,q):
        train_arima = train_data['Close']
        test_arima = test_data['Close']
        history = [x for x in train_arima]
        y = test_arima
        # make first prediction
        predictions = list()
        model = sm.tsa.arima.ARIMA(history, order=(p,1,q))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(y[0])
        # rolling forecasts
        for i in range(1, len(y)):
            # predict
            model = sm.tsa.arima.ARIMA(history, order=(p,d,q))
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            # invert transformed prediction
            predictions.append(yhat)
            # observation
            obs = y[i]
            history.append(obs)
        
        return predictions, y