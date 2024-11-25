import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import json
import os

# Function to create sequences for prediction
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Function to load model and predict future prices
def predict_future_prices(ticker, seq_length=30, future_days=30):
    # Load the model
    model_filename = f"../models/LSTM_{ticker}.pkl"
    with open(model_filename, "rb") as file:
        model = pickle.load(file)

    # Load the latest stock data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Ensure data contains the Close column
    if 'Close' not in data.columns:
        raise ValueError(f"Data for {ticker} does not contain 'Close' prices.")

    close_prices = data['Close'].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Use the last seq_length data points for prediction
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []

    for _ in range(future_days):
        next_pred = model.predict(last_sequence[np.newaxis, :, :])
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

    # Transform predictions back to original scale
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Generate dates for the next 30 days
    future_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(future_days)]

    return future_dates, future_prices

# Main function
def main():
    tickers = ['ADRO.JK', 'BBCA.JK', 'TLKM.JK']
    seq_length = 30
    future_days = 30

    future_predictions_all = {}
    csv_data = []

    for ticker in tickers:
        print(f"Predicting future prices for {ticker}...")
        try:
            future_dates, future_prices = predict_future_prices(ticker, seq_length, future_days)
            future_predictions_all[ticker] = {
                "dates": future_dates,
                "prices": future_prices.tolist()
            }

            # Save data to CSV format
            csv_file = f"../predictions/LSTM_{ticker}_Future_Predictions.csv"
            df = pd.DataFrame({"Date": future_dates, "Predicted Price (IDR)": future_prices})
            df.to_csv(csv_file, index=False)
            print(f"Future predictions for {ticker} saved to {csv_file}")

            # Add to combined CSV data
            csv_data.append(pd.DataFrame({"Ticker": ticker, "Date": future_dates, "Predicted Price (IDR)": future_prices}))

            # Plot predicted future prices
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_prices, marker='o', label="Predicted Prices")
            plt.title(f"Predicted Future Prices for {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Price (IDR)")
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../predictions/LSTM_{ticker}_Future_Predictions.png")
            plt.show(block=False)
            plt.pause(0.5)  # Small pause to allow the plot to render
        except Exception as e:
            print(f"Error predicting prices for {ticker}: {e}")

    # Save future predictions to JSON
    with open("../predictions/LSTM_Future_Predictions.json", "w") as json_file:
        json.dump(future_predictions_all, json_file, indent=4)

    # Save combined CSV
    combined_csv_file = "../predictions/LSTM_Future_Predictions_Combined.csv"
    combined_df = pd.concat(csv_data, ignore_index=True)
    combined_df.to_csv(combined_csv_file, index=False)
    print(f"\nCombined future predictions saved to {combined_csv_file}")

if __name__ == "__main__":
    main()
