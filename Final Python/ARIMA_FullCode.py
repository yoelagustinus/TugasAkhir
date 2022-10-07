import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import statsmodels.api as sm
from matplotlib.pylab import rcParams
# from statsmodels.api.tsa.arima_model import ARIMA
from math import sqrt
import yfinance as yf
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import mysql.connector as mysql


#hyperparameters
arr_p  = [1,2]
arr_q = [1,2]
start_date = "2017-01-01"
d = 1

arr_end_date = ["2021-12-31","2017-12-31", "2017-03-31"]
arr_symbol_dataset = ["GGRM.jk","UNVR.jk","PSDN.jk"]
column_dataset_obs = 'Close'


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features

# Rmse, mae, mape
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return math.sqrt(mean_error)

def mae_metric(actual, predicted):
    y_true, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(actual - predicted))
    
def mape_metric(actual, predicted): 
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

jumlah_pengujian = 0

for index_dataset in arr_symbol_dataset:
    for index_end_date in arr_end_date:
        for p in arr_p:
            for q in arr_q: 
                jumlah_pengujian+=1
                data = yf.download(index_dataset, start=start_date, end=index_end_date)


                if data.shape[0]>=1250:
                    term_status = "long"
                elif data.shape[0]>=250:
                    term_status = "mid"
                else:
                    term_status = "short"
                    
                # Splitting Data
                train_data, test_data = data[0:int((len(data)-1)*0.8)], data[int((len(data)+1)*0.8):]

                #ARIMA Model
                train_arima = train_data[column_dataset_obs]
                test_arima = test_data[column_dataset_obs]

                history = [x for x in train_arima]
                y = test_arima
                # make first prediction
                predictions = list()
                model = sm.tsa.arima.ARIMA(history, order=(p,d,q))
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

                # Plotting Splitting Training, Testing and Save
                plt.figure(figsize=(16,8))
                plt.plot(data.index[:], data[column_dataset_obs], color='green', label = 'Train Stock Price')
                plt.plot(test_data.index, y, color = 'red', label = 'Real Stock Price')
                plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
                plt.title(index_dataset+ ' - ' + term_status +  ' Stock Prediction, p: '+ str(p) +'; q: '+str(q))
                plt.xlabel('Time')
                plt.ylabel(index_dataset +' Stock Price '+ column_dataset_obs)
                plt.grid(True)
                plt.savefig("../results/ARIMA/plots/" + index_dataset +'_ARIMA-'+ 
                                            term_status + '_p='+ str(p) +'_q='+ str(q) + '.pdf')
                
                price_prediction=predictions
                new_date = pd.to_datetime(test_data.index)

                # Save as new Dataset
                new_data = {'Date': new_date,
                'real_close': y,
                'close_arima': price_prediction}
                
                df_new_data = pd.DataFrame(new_data, columns = ['Date', 'real_close','close_arima'])

                df_new_data.to_csv("../results/ARIMA/datasets/" + term_status + "/"+ index_dataset
                +'_ARIMA-'+ term_status + '_p='+ str(p) +'_q='+ str(q) + '.csv', index=False)


                # Report Performance of ARIMA Predictions
                print('P: ' + str(p))
                print('D: ' + str(d))
                print('Q: ' + str(q))
                rmse = rmse_metric(y, predictions)
                rmse = np.round(rmse, 2)
                print(f'Root Mean Square Error (RMSE): {rmse}')

                mae = mae_metric(y, predictions)
                mae = np.round(mae, 2)
                print(f'Median Absolute Error (MAE): {mae}')

                mape = mape_metric(y, predictions)
                mape = mape*100
                mape = np.round(mape, 2)
                print(f'Mean Absolute Percentage Error (MAPE): {mape} %')




                obs_dataset = index_dataset+'-'+term_status

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