# coding=UTF-8
'''
@Date    : 2022.05.29
@Author  : Jethro
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models import modelss
from Decomposition import Decomposition
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from keras_tuner import RandomSearch
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, LeakyReLU

warnings.filterwarnings('ignore')

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(series):
    statistic, p_value, lags, critical_values = kpss(series, regression='c')
    print('KPSS Statistic: %f' % statistic)
    print('p-value: %f' % p_value)
    print('Critical Values:')
    for key, value in critical_values.items():
        print('\t%s: %.3f' % (key, value))

def plot_acf_pacf_seasonal(series):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # ACF Plot
    axs[0, 0].set_title('ACF Plot')
    plot_acf(series, ax=axs[0, 0], lags=40)

    # PACF Plot
    axs[0, 1].set_title('PACF Plot')
    plot_pacf(series, ax=axs[0, 1], lags=40)

    # Seasonal Decomposition
    decomposition = seasonal_decompose(series, model='additive', period=12)  # Adjust period as needed
    
    # Plot each component separately
    axs[1, 0].set_title('Seasonal Decomposition')
    axs[1, 0].plot(decomposition.trend, label='Trend', color='orange')
    axs[1, 0].plot(decomposition.seasonal, label='Seasonal', color='blue')
    axs[1, 0].plot(decomposition.resid, label='Residual', color='green')
    axs[1, 0].legend()

    # Trend Plot
    axs[1, 1].set_title('Trend Plot')
    axs[1, 1].plot(decomposition.trend, label='Trend', color='orange')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def denoise(data, imfs):
    data = data.reshape(-1)
    denoise_data = 0
    for i in range(imfs.shape[1]):
        denoise_data += imfs[:,i]
        pearson_corr_coef = np.corrcoef(denoise_data, data)
        if pearson_corr_coef[0,1] >=0.995:
            print(i)
            break

    return denoise_data

def IMF_decomposition(data, length):
    Decomp = Decomposition(data, length)
    ssa_imfs  = Decomp.SSA()
    ssa_denoise = denoise(data, ssa_imfs)
    return ssa_denoise

def Data_partitioning(data,test_number, input_step, pre_step):
    dataset = data.reshape(-1,1)
    scaled_tool = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaled_tool.fit_transform(dataset)
    step_size = input_step
    data_input= np.zeros((len(data_scaled) - step_size - pre_step, step_size))
    data_label = np.zeros((len(data_scaled) - step_size - pre_step, 1))
    for i in range(len(data_scaled) - step_size-pre_step):
        data_input[i, :] = data_scaled[i:step_size + i,0]
        data_label[i, 0] = data_scaled[step_size + i + pre_step,0]
    X_train = data_input[:-test_number]
    Y_train = data_label[:-test_number]
    X_test = data_input[-test_number:]
    Y_test = data_label[-test_number:]

    return  X_train, X_test, Y_train, Y_test, scaled_tool

def single_model(data,test_number,flag, input_step, pre_step):
    X_train, X_test, Y_train, Y_test, scaled_tool = Data_partitioning(data, test_number, input_step, pre_step)
    model = modelss(X_train, X_test, Y_train, Y_test, scaled_tool)
    if flag == 'tcn_gru':
        pre = model.run_tcn_gru()
    if flag == 'tcn_lstm':
        pre = model.run_tcn_lstm()
    if flag == 'tcn_rnn':
        pre = model.run_tcn_rnn()
    if flag == 'tcn_bpnn':
        pre = model.run_tcn_bpnn()
    if flag == 'gru':
        pre = model.run_GRU()
    if flag == 'lstm':
        pre = model.run_LSTM()
    if flag == 'rnn':
        pre = model.run_RNN()
    if flag == 'bpnn':
        pre = model.run_BPNN()
    data_pre = pre[:, 0]

    return data_pre

# Define a function to build the model
def build_model(hp, input_step):
    model = Sequential()
    model.add(Input(batch_shape=(None, input_step, 1)))
    model.add(TCN(nb_filters=hp.Int('nb_filters', 8, 64, step=8), kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
    model.add(GRU(units=hp.Int('units', 16, 128, step=16), return_sequences=False))
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    test_number, imfs_number, input_step, pre_step= 200, 15, 20, 2
    data = pd.read_csv('10 min wind speed data.csv', header=None)
    ssa_denoise = IMF_decomposition(data.iloc[:,2].values, imfs_number)
    
    # Perform ADF and KPSS tests
    print("ADF Test Results:")
    adf_test(ssa_denoise)
    print("\nKPSS Test Results:")
    kpss_test(ssa_denoise)

    # Plot ACF, PACF, and Seasonal Decomposition
    plot_acf_pacf_seasonal(ssa_denoise)

    np.savetxt('ssa_denoise_3.csv', ssa_denoise[-test_number:], delimiter=',')
    
    # Data partitioning
    X_train, X_test, Y_train, Y_test, scaled_tool = Data_partitioning(ssa_denoise, test_number, input_step, pre_step)

    # Define Actual values for comparison
    Actual = ssa_denoise[-test_number:]  # Define Actual here

    # Hyperparameter tuning
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_step),
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld'
    )

    # Assuming you have your data prepared
    tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Model predictions after tuning
    pre_ssa_tcn_gru = single_model(ssa_denoise, test_number, 'tcn_gru', input_step, pre_step)
    pre_ssa_tcn_lstm = single_model(ssa_denoise, test_number, 'tcn_lstm', input_step, pre_step)
    pre_ssa_tcn_rnn = single_model(ssa_denoise, test_number, 'tcn_rnn', input_step, pre_step)
    pre_ssa_tcn_bpnn = single_model(ssa_denoise, test_number, 'tcn_bpnn', input_step, pre_step)
    pre_ssa_gru = single_model(ssa_denoise, test_number, 'gru', input_step, pre_step)
    pre_ssa_lstm = single_model(ssa_denoise, test_number, 'lstm', input_step, pre_step)
    pre_ssa_rnn = single_model(ssa_denoise, test_number, 'rnn', input_step, pre_step)
    pre_ssa_bpnn = single_model(ssa_denoise, test_number, 'bpnn', input_step, pre_step)

    # Plotting all model predictions vs Actual
    plt.figure(figsize=(15, 20))

    # TCN GRU
    plt.subplot(4, 2, 1)
    plt.plot(pre_ssa_tcn_gru, color='black', label='Predicted TCN GRU')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('TCN GRU Predictions vs Actual')
    plt.legend()

    # TCN LSTM
    plt.subplot(4, 2, 2)
    plt.plot(pre_ssa_tcn_lstm, color='m', label='Predicted TCN LSTM')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('TCN LSTM Predictions vs Actual')
    plt.legend()

    # TCN RNN
    plt.subplot(4, 2, 3)
    plt.plot(pre_ssa_tcn_rnn, color='y', label='Predicted TCN RNN')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('TCN RNN Predictions vs Actual')
    plt.legend()

    # TCN BPNN
    plt.subplot(4, 2, 4)
    plt.plot(pre_ssa_tcn_bpnn, color='red', label='Predicted TCN BPNN')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('TCN BPNN Predictions vs Actual')
    plt.legend()

    # GRU
    plt.subplot(4, 2, 5)
    plt.plot(pre_ssa_gru, color='green', label='Predicted GRU')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('GRU Predictions vs Actual')
    plt.legend()

    # LSTM
    plt.subplot(4, 2, 6)
    plt.plot(pre_ssa_lstm, color='purple', label='Predicted LSTM')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('LSTM Predictions vs Actual')
    plt.legend()

    # RNN
    plt.subplot(4, 2, 7)
    plt.plot(pre_ssa_rnn, color='orange', label='Predicted RNN')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('RNN Predictions vs Actual')
    plt.legend()

    # BPNN
    plt.subplot(4, 2, 8)
    plt.plot(pre_ssa_bpnn, color='cyan', label='Predicted BPNN')
    plt.plot(Actual, color='blue', label='Actual')
    plt.title('BPNN Predictions vs Actual')
    plt.legend()

    plt.tight_layout()
    plt.show()
