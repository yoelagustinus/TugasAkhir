import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt

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


#hyperparameters
arr_p  = [1,2]
arr_q = [1,2]
start_date = "2017-01-01"
d = 1

arr_end_date = ["2021-12-31","2017-12-31", "2017-03-31"]
arr_symbol_dataset = ["GGRM.jk","UNVR.jk","PSDN.jk"]
column_dataset_obs = 'Close'

jumlah_pengujian = 0


for symbol_dataset in arr_symbol_dataset:
    for end_date in arr_end_date:
        for p in arr_p:
            for q in arr_q:
                
                jumlah_pengujian += 1

                df = DataLoad.read_data(start_date, end_date, symbol_dataset)

                if df.shape[0]>=1250:
                    term_status = "long"
                elif df.shape[0]>=250:
                    term_status = "mid"
                else:
                    term_status = "short"
                    
                train_data, test_data = Preprocessing.splitting_dataset(df)

                predictions, y = ARIMA_model.forecast_model(train_data, test_data, p,q)

                plt.figure(figsize=(16,8))
                plt.plot(df.index[:], df[column_dataset_obs], color='green', label = 'Train Stock Price')
                plt.plot(test_data.index, y, color = 'red', label = 'Real Stock Price')
                plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
                plt.title(symbol_dataset+ ' - ' + term_status +  ' Stock Prediction, p: '+ str(p) +'; q: '+str(q))
                plt.xlabel('Time')
                plt.ylabel(symbol_dataset +' Stock Price '+ column_dataset_obs)
                plt.legend()
                plt.grid(True)
                #plt.savefig("../results/ARIMA/plots/" + symbol_dataset +'_ARIMA-'+ term_status + '_p='+ str(p) +'_q='+ str(q) + '.pdf')
                
                plt.figure()
                plt.plot(test_data.index, y, color = 'red', label = 'Real Stock Price')
                plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
                plt.title(symbol_dataset+ ' - ' + term_status +' Stock Prediction, p: '+ str(p) +'; q: '+str(q))
                plt.xlabel('Time')
                plt.ylabel(symbol_dataset +' Stock Price '+ column_dataset_obs)
                plt.legend()
                plt.grid(True)
                
                # mse = mean_squared_error(y, predictions)
                # print('MSE: '+str(mse))
                print(symbol_dataset+"-"+term_status)
                print('P: ' + str(p))
                print('D: ' + str(d))
                print('Q: ' + str(q))

                rmse = EvaluasiForecasting.rmse_metric(y, predictions)
                rmse = np.round(rmse, 2)
                print(f'Root Mean Square Error (RMSE): {rmse}')

                mae = EvaluasiForecasting.mae_metric(y, predictions)
                mae = np.round(mae, 2)
                print(f'Median Absolute Error (MAE): {mae}')

                mape = EvaluasiForecasting.mape_metric(y, predictions)
                mape = mape*100
                mape = np.round(mape, 2)
                print(f'Mean Absolute Percentage Error (MAPE): {mape} %')

                price_prediction=predictions
                new_date = pd.to_datetime(test_data.index)

                new_data = {'Date': new_date,
                    'real_close': y,
                'close_arima': price_prediction}

                df_new_data = pd.DataFrame(new_data, columns = ['Date', 'real_close','close_arima'])

                #df_new_data.to_csv("../results/ARIMA/datasets/" + term_status + "/"+ symbol_dataset + '_ARIMA-'+ term_status + '_p='+ str(p) +'_q='+ str(q) + '.csv', index=False)

                

                obs_dataset = symbol_dataset+'-'+term_status

                #connect database
                mydb = mysql.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="db_tugasakhir"
                )
                mycursor = mydb.cursor()

                #insert to database
                sql = "INSERT INTO pengujian_arima (datasets, start_dates, end_dates,p, d, q, RMSE, MAE, MAPE) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                val = (obs_dataset, start_date, index_end_date,p, d, q, rmse, mae, mape)

                mycursor.execute(sql,val)
                mydb.commit()
                print("pengujian ke: " + str(jumlah_pengujian))
                print("=================================================================")
