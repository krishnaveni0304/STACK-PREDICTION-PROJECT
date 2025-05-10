import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download stock data
def fetch_stock_data(ticker='AAPL', period='5y'):
    data = yf.download(ticker, period=period)
    # Check if the data is empty and handle the error
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}' and period '{period}'. "
                         f"Check the ticker symbol and period, and ensure you have an internet connection.")
    return data['Close']

# Step 2: Preprocess data
def preprocess_data(data, look_back=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Step 3: Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Step 4: Train model
def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Step 5: Predict and plot
def predict_and_plot(model, X, y, scaler, original_data):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    plt.figure(figsize=(12,6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Stock Price Prediction vs Actual')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    print("\n📈 Last Actual Price:", actual[-1][0])
    print("🔮 Next Predicted Price:", predictions[-1][0])

# Main Program
def main():
    print("Fetching data and training model...")
    try:
        close_prices = fetch_stock_data('AAPL', '5y')
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    look_back = 10
    X, y, scaler = preprocess_data(close_prices, look_back)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model((X.shape[1], 1))
    model = train_model(model, X, y)

    predict_and_plot(model, X, y, scaler, close_prices)

if _name_ == "_main_":
    main()
