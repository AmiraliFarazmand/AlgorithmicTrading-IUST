import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()





# Plot ACF
def acf_plt(data):
    plot_acf(data)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

# Plot PACF
def dacf_plt(data):
    plot_pacf(data)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

def fetch_crypto_data(symbol, start_date, end_date, priod):
    data = yf.Ticker(symbol)
    
    tickerDf = data.history(period=priod, start=start_date, end=end_date)
    tickerDf = tickerDf[['Close']]
    print(tickerDf)   
    
    return tickerDf

def get_data_and_plot(crypto, timeframe,start,end):
    data = fetch_crypto_data(crypto, start, end, timeframe)
    acf_plt(data['Close'])
    dacf_plt(data['Close'])
    return data

def predict_and_plot(data:pd.DataFrame, train_end, test_end, p=0,q=0,plot_option=False):
    train_data = data.loc[:train_end]#.loc???
    test_data = data.loc[train_end + timedelta(days=1):test_end]
    
    model = ARIMA(train_data, order=(p,0,q))
    model_fit = model.fit()
    print(model_fit.summary())
    
    pred_start_date = test_data.index[0]
    pred_end_date = test_data.index[-1]
    predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
    # predictions = model_fit.predict(start=start_date, end=train_end)
    # test_data = test_data.dropna()
    # predictions = predictions.dropna()

    residuals = test_data['Close'] - predictions

    # print(test_data,'\n\\\\\\\\\\\\\\\\')
    # print(predictions,'\n/////////////')

    # print(residuals)
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data.Close)),8))
    
    # epsilon = 1e-10  # small constant
    # mape = np.mean(np.abs((residuals + epsilon) / (test_data['Close'] + epsilon)))
    # print('Mean Absolute Percent Error:', round(mape, 8))
    print('Root Mean Squared Error:', round(np.sqrt(np.mean(residuals**2)),8))
    # print(test_data['Close'].index)
    # print(predictions.index)

    if plot_option:
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index, test_data['Close'], label='Actual')

        # Plot the predicted values
        plt.plot(predictions.index, predictions, label='Predicted')

        # Set the title and labels
        plt.title(f'ARIMA Model Predictions. ARMA({p},{q})')
        plt.xlabel('Date')
        plt.ylabel('Close Price')

        # Show the legend
        plt.legend()

        # Show the plot
        plt.show()
        # Plot the entire year's data
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Actual')

        # Plot the predicted values
        plt.plot(predictions.index, predictions, label='Predicted', color='red')

        # Set the title and labels
        plt.title(f'ARIMA Model Predictions. ARMA({p},{q})')
        plt.xlabel('Date')
        plt.ylabel('Close Price')

        # Show the legend
        plt.legend()

        # Show the plot
        plt.show()
    

def get_mse_msp(data:pd.DataFrame,train_data, test_data, train_end, test_end,p,q):
    model = ARIMA(train_data, order=(p,0,q))
    model_fit = model.fit()
    # print(model_fit.summary())
    
    pred_start_date = test_data.index[0]
    pred_end_date = test_data.index[-1]
    predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
    residuals = test_data['Close'] - predictions
    # temp_mape = round(np.mean(abs(residuals/test_data.Close)),8)
    # temp_mse = round(np.sqrt(np.mean(residuals**2)),8)
    temp_mape = np.mean(abs(residuals/test_data.Close))
    temp_mse = np.sqrt(np.mean(residuals**2))
    
    return (temp_mape, temp_mse)
  
def get_best_combination(data:pd.DataFrame, train_end, test_end, def_p=0, def_q=0, loop_over_p=False, loop_over_q=False, max_loop = 5):

    mse, mape =99999,999999
    best_combination = {'p':def_p, 'q':def_q}
    train_data = data.loc[:train_end]#.loc???
    test_data = data.loc[train_end + timedelta(days=1):test_end]
    if loop_over_p and loop_over_q: 
        for p in range(max_loop):
            for q in range(max_loop):
               
                temp_mape, temp_mse = get_mse_msp(data, train_data, test_data, train_end,test_end, p,q) 
                print(f'ARMA({p},{q}) -> MSE={temp_mse}, MAPE={temp_mape}')
                # if  temp_mse <=mse: #and temp_mape <= mape 
                if np.less(temp_mse, mse):
                    mse = temp_mse
                    best_combination['p'] = p
                    best_combination['q'] = q

    elif loop_over_p:
        for p in range(max_loop):
            
            temp_mape, temp_mse = get_mse_msp(data, train_data, test_data, train_end,test_end, p,def_q)
            print(f'ARMA({p},{def_q}) -> MSE={temp_mse}, MAPE={temp_mape}')
            # if  temp_mse <=mse:#temp_mape <= mape 
            if np.less(temp_mse, mse):
                mse = temp_mse
                best_combination['p'] = p
                # best_combination['q'] = def_q

    elif loop_over_q:
        for q in range(max_loop):
            temp_mape, temp_mse = get_mse_msp(data, train_data, test_data, train_end,test_end, def_p,q)
            print(f'ARMA({def_p},{q}) -> MSE={temp_mse}, MAPE={temp_mape}')
            # if  temp_mse <=mse: #temp_mape <= mape 
            if np.less(temp_mse, mse):
                mse = temp_mse
                # best_combination['p'] = def_p
                best_combination['q'] = q
    else: 
        raise ValueError("looped not over p neither q, or something else")
    return best_combination







# ==================================

start_date = datetime(2022, 11, 1, tzinfo=timezone('Asia/Tehran'))
end_date = datetime(2023, 11, 1, tzinfo=timezone('Asia/Tehran'))
train_end = datetime(2023,10,1, tzinfo=timezone('Asia/Tehran'))
test_end = datetime(2023,11,1, tzinfo=timezone('Asia/Tehran') )
data = get_data_and_plot('USDT-USD', '1d', start_date,end_date)

best_comb = get_best_combination(data, train_end,test_end,loop_over_p=True,loop_over_q=True ,max_loop=20)
print(best_comb)
predict_and_plot(data, train_end,test_end,p= best_comb['p'],q = best_comb['q'],plot_option=True)