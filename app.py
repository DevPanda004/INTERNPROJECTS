import os
from flask import Flask, render_template, request, send_from_directory
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

app = Flask(__name__)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=100))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock']
        end_date = request.form['end_date']
        transaction_cost = float(request.form.get('transaction_cost', 0.001))  # Default transaction cost

        # Load data
        data = yf.download(stock_name, start='2010-01-01', end=end_date)
        data['Close'] = data['Adj Close']
        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        seq_length = 60
        train_data = scaled_data[:int(len(scaled_data) * 0.8)]
        test_data = scaled_data[int(len(scaled_data) * 0.8) - seq_length:]
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)

        # Train the model
        model = build_model(seq_length)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_inverse = scaler.inverse_transform(y_pred)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
        # Create a DataFrame to compare the predicted vs actual values
        comparison_df = pd.DataFrame({
            'Actual': y_test_inverse.flatten()[-548:],
            'Predicted': y_pred_inverse.flatten()[-548:]
        })


        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-len(comparison_df):], comparison_df['Actual'], color='blue', label='Actual Stock Price')
        plt.plot(data.index[-len(comparison_df):], comparison_df['Predicted'], color='red', label='Predicted Stock Price')
        plt.title(f'{stock_name} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', plot_url=plot_path)

    return render_template('index.html', plot_url=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
