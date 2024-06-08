import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# Load data
data = yf.download('AAPL', start='2010-01-01', end='2020-01-01')

# Preprocess data
data['Close'] = data['Adj Close']
data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60

# Split data into training and test sets using indices
train_data = scaled_data[:int(len(scaled_data) * 0.8)]  # First 80% of the data for training
test_data = scaled_data[int(len(scaled_data) * 0.8) - seq_length:]  # Last 20% for testing

# Create train and test sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define custom callback to print accuracy
class PrintAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}, Loss: {logs['loss']}, Accuracy: {logs.get('accuracy', 'N/A')}")

# Create the model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[PrintAccuracy()])

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_inverse = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a DataFrame to compare the predicted vs actual values
comparison_df = pd.DataFrame({
    'Actual': y_test_inverse.flatten(),
    'Predicted': y_pred_inverse.flatten()
})

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['Actual'], color='blue', label='Actual Stock Price')
plt.plot(comparison_df['Predicted'], color='red', label='Predicted Stock Price')
plt.title('AAPL Stock Price Prediction (2018-2020)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Optionally, save the comparison DataFrame to a CSV file
comparison_df.to_csv('predicted_vs_actual_2020_2022.csv', index=False)
