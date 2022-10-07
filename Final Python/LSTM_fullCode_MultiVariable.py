import math 
import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime 
from pandas.plotting import register_matplotlib_converters 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import seaborn as sns 
import mysql.connector as mysql
import yfinance as yf

#hyperparameters
arr_epochs  = [10,100,1000]
arr_units = [10,50,128]
start_date = "2017-01-01"

arr_end_date = ["2021-12-31","2017-12-31", "2017-03-31"]
arr_symbol_dataset = ["GGRM.jk","UNVR.jk","PSDN.jk"]


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

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
        for epoch in arr_epochs:
            for unit in arr_units: 
                
                jumlah_pengujian+=1
                df = yf.download(index_dataset, start=start_date, end=index_end_date)
                if df.shape[0]>=1250:
                    term_status = "long"
                elif df.shape[0]>=250:
                    term_status = "mid"
                else:
                    term_status = "short"

                # Indexing Batches
                train_df = df.sort_values(by=['Date']).copy()

                # Daftar Fitur yang digunakan
                FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']

                print('FEATURE LIST')
                print([f for f in FEATURES])

                # Buat kumpulan data dengan fitur dan filter data ke daftar FEATURES
                data = pd.DataFrame(train_df)
                data_filtered = data[FEATURES]

                # Kami menambahkan kolom prediksi dan menetapkan nilai dummy untuk menyiapkan data untuk penskalaan
                data_filtered_ext = data_filtered.copy()
                data_filtered_ext['Prediction'] = data_filtered_ext['Close']

                # Dapatkan jumlah baris dalam data
                nrows = data_filtered.shape[0]
                # print(nrows)

                # Convert the data ke numpy values
                np_data_unscaled = np.array(data_filtered)
                # print(np_data_unscaled)
                np_data = np.reshape(np_data_unscaled, (nrows, -1))
                # print(np_data.shape)

                # Transform the data by scaling each feature to a range between 0 and 1 using MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0,1))
                np_data_scaled = scaler.fit_transform(np_data_unscaled)

                # Membuat scaler terpisah yang berfungsi pada satu kolom untuk prediksi penskalaan
                scaler_pred = MinMaxScaler(feature_range=(0,1))
                df_Close = pd.DataFrame(data_filtered_ext['Close'])
                np_Close_scaled = scaler_pred.fit_transform(df_Close)

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
                x_train, y_train = partition_dataset(sequence_length, train_data)
                x_test, y_test = partition_dataset(sequence_length, test_data)

                # Configure the neural network model
                model = Sequential()
                model.add(Bidirectional(LSTM(unit, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]))))
                model.add(Dense(1))


                # Compile the model
                model.compile(optimizer='adam', loss='mse')
                # Training the model
                early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
                history = model.fit(x_train, y_train, batch_size=16, epochs=epoch,validation_data=(x_test, y_test))

                # Get the predicted values
                y_pred_scaled = model.predict(x_test)

                # Unscale the predicted values
                y_pred = scaler_pred.inverse_transform(y_pred_scaled)
                y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

                # Root Mean Square Error (RMSE)
                RMSE = rmse_metric(y_test_unscaled, y_pred)
                RMSE = np.round(RMSE, 2)
                print(f'Root Mean Square Error (RMSE): {RMSE}')

                # Mean Absolute Error (MAE)
                MAE = mae_metric(y_test_unscaled, y_pred)
                MAE = np.round(MAE, 2)
                print(f'Median Absolute Error (MAE): {MAE}')

                # Mean Absolute Percentage Error (MAPE)
                MAPE = mape_metric(y_test_unscaled, y_pred)
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
                plt.savefig("../results/LSTM2/plots/" + index_dataset +'_LSTM-'+ term_status + '_e='+ str(epoch) +'_u='+ str(unit) + '.pdf')
                plt.legend()

                #save to new dataset
                new_data = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'real_close'})
                new_data['close_lstm'] = y_pred
                df_new_data = pd.DataFrame(new_data)
                df_new_data.to_csv("../results/LSTM2/datasets/" + index_dataset +'_LSTM-'+ term_status + '_e='+ str(epoch) +'_u='+ str(unit) + '.csv', index=True)

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
                sql = "INSERT INTO pengujian_lstm2 (datasets, start_dates, end_dates,epochs, units, RMSE, MAE, MAPE) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                val = (obs_dataset, start_date, index_end_date, epoch, unit, RMSE, MAE, MAPE)

                mycursor.execute(sql,val)
                mydb.commit()
                print("pengujian ke: " + str(jumlah_pengujian))
                print("=================================================================")