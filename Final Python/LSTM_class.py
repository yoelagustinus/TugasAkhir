import math 
import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime 
from pandas.plotting import register_matplotlib_converters 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import seaborn as sns 
import mysql.connector as mysql
import yfinance as yf
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
    def feature_selection(df):
        # Indexing Batches
        train_df = df.sort_values(by=['Date']).copy()

        # Daftar Fitur yang digunakan
        FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']
        data = pd.DataFrame(train_df)
        data_filtered = data[FEATURES]

        # menambahkan kolom prediksi dan menetapkan nilai dummy untuk menyiapkan data untuk penskalaan
        data_filtered_ext = data_filtered.copy()
        data_filtered_ext['Prediction'] = data_filtered_ext['Close']
        return data_filtered_ext, data_filtered, data
    
    def reshape_data(data_filtered):
        # Dapatkan jumlah baris dalam data
        nrows = data_filtered.shape[0]

        # Convert the data ke numpy values
        np_data_unscaled = np.array(data_filtered)
        np_data = np.reshape(np_data_unscaled, (nrows, -1))
        
        return np_data_unscaled, np_data

    def min_max(np_data_unscaled,data_filtered_ext):
        scaler = MinMaxScaler(feature_range=(0,1))
        np_data_scaled = scaler.fit_transform(np_data_unscaled)

        # Membuat scaler terpisah yang berfungsi pada satu kolom untuk prediksi penskalaan
        scaler_pred = MinMaxScaler(feature_range=(0,1))
        df_Close = pd.DataFrame(data_filtered_ext['Close'])
        np_Close_scaled = scaler_pred.fit_transform(df_Close)
        
        return np_data_scaled, np_Close_scaled, scaler_pred
    
    def inverse_minmax(y_pred_scaled, y_test):
        y_pred = scaler_pred.inverse_transform(y_pred_scaled)
        y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
        
        return y_pred, y_test_unscaled
    
    def partition_dataset(sequence_length, data, index_Close):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction

        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y
    
    def splitting_dataset(np_data_scaled,data):
        # Set the sequence length - this is the timeframe used to make a single prediction
        sequence_length = 1

        # Prediction Index
        index_Close = data.columns.get_loc("Close")

        # Split the training data into train and train data sets
        # As a first step, we get the number of rows to train the model on 80% of the data 
        train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

        # Create the training and test data
        train_data = np_data_scaled[0:train_data_len, :]
        test_data = np_data_scaled[train_data_len - sequence_length:, :]
        
        # Generate training data and test data
        x_train, y_train = Preprocessing.partition_dataset(sequence_length, train_data, index_Close)
        x_test, y_test = Preprocessing.partition_dataset(sequence_length, test_data, index_Close)
        
        return x_train, y_train, x_test, y_test, train_data_len
class LSTM_unit:
    def training_model(x_train, y_train, x_test, y_test, unit, epoch):
        # Configure the neural network model
        model = Sequential()
        model.add(Bidirectional(LSTM(unit, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]))))
        model.add(Dense(1))

        # Compile the model
        model.compile(loss='mse')
        # Training the model
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history = model.fit(x_train, y_train, 
                batch_size=16, 
                epochs=epoch,
                validation_data=(x_test, y_test))
        return x_test, model
        
    def predict_model(x_test,model):
        y_pred_scaled = model.predict(x_test)
        return y_pred_scaled


#hyperparameters
arr_epochs  = [10,100,1000]
arr_units = [10,50,128]
start_date = "2017-01-01"

arr_end_date = ["2017-03-31", "2017-12-31", "2021-12-31"]
arr_symbol_dataset = ["GGRM.jk", "UNVR.jk", "PSDN.jk"]

jumlah_pengujian = 0
for symbol_dataset in arr_symbol_dataset:
    for end_date in arr_end_date:
        for epoch in arr_epochs:
            for unit in arr_units: 
                
                jumlah_pengujian+=1

                # load the time series Data use Yahoo Finance
                df = DataLoad.read_data(start_date, end_date,symbol_dataset)

                if df.shape[0]>=1250:
                    term_status = "long"
                elif df.shape[0]>=250:
                    term_status = "mid"
                else:
                    term_status = "short"

                #feature selection and scaling
                data_filtered_ext, data_filtered, data = Preprocessing.feature_selection(df)
                np_data_unscaled, np_data = Preprocessing.reshape_data(data_filtered)
                np_data_scaled, np_Close_scaled, scaler_pred = Preprocessing.min_max(np_data_unscaled,data_filtered_ext)

                #split train and test
                x_train, y_train, x_test, y_test, train_data_len = Preprocessing.splitting_dataset(np_data_scaled,data) 

                # Train the Multivariable Prediction Model
                x_test,model=LSTM_unit.training_model(x_train, y_train, x_test, y_test, unit, epoch)

                # Predict data using data test
                y_pred_scaled = LSTM_unit.predict_model(x_test,model)

                #inverse minmax
                y_pred, y_test_unscaled = Preprocessing.inverse_minmax(y_pred_scaled, y_test)

                # Evaluate model performance
                # Root Mean Square Error (RMSE)
                RMSE = EvaluasiForecasting.rmse_metric(y_test_unscaled, y_pred)
                RMSE = np.round(RMSE, 2)
                print(f'Root Mean Square Error (RMSE): {RMSE}')

                # Mean Absolute Error (MAE)
                MAE = EvaluasiForecasting.mae_metric(y_test_unscaled, y_pred)
                MAE = np.round(MAE, 2)
                print(f'Median Absolute Error (MAE): {MAE}')

                # Mean Absolute Percentage Error (MAPE)
                MAPE = EvaluasiForecasting.mape_metric(y_test_unscaled, y_pred)
                MAPE = np.round(MAPE, 2)
                print(f'Mean Absolute Percentage Error (MAPE): {MAPE} %')

                #save plot
                # The date from which on the date is displayed
                display_start_date = start_date

                # Add the difference between the valid and predicted prices
                train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
                valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
                valid.insert(1, "y_pred", y_pred, True)
                # valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
                df_union = pd.concat([train, valid])

                # Zoom in to a closer timeframe
                df_union_zoom = df_union[df_union.index > display_start_date]

                # Create the lineplot
                fig, ax1 = plt.subplots(figsize=(16, 8))
                plt.title("Predict Data vs Test Data")

                sns.set_palette(["#FF0000", "#1960EF", "#00FF00"])
                sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
                plt.savefig("../results/LSTM/plots/" + symbol_dataset +'_LSTM-'+ term_status + '_e='+ str(epoch) +'_u='+ str(unit) + '.pdf')
                plt.legend()

                #save to new dataset
                new_data = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'real_close'})
                new_data['close_lstm'] = y_pred
                df_new_data = pd.DataFrame(new_data)
                df_new_data.to_csv("../results/LSTM/datasets/" + symbol_dataset +'_LSTM-'+ term_status + '_e='+ str(epoch) +'_u='+ str(unit) + '.csv', index=True)

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
                sql = "INSERT INTO pengujian_lstm3 (datasets, start_dates, end_dates,epochs, units, RMSE, MAE, MAPE) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                val = (obs_dataset, start_date, end_date, epoch, unit, RMSE, MAE, MAPE)

                mycursor.execute(sql,val)
                mydb.commit()
                print("pengujian ke: " + str(jumlah_pengujian))
                print("=================================================================")