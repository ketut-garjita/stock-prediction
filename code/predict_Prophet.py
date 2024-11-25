import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os
import json

# Function to load Prophet model and make predictions
def load_and_predict(ticker, future_days=30):
    # Load the saved Prophet model
    model_filename = f"../models/Prophet_{ticker}.pkl"
    with open(model_filename, "rb") as file:
        model = pickle.load(file)

    # Create a dataframe for future predictions
    future = model.make_future_dataframe(periods=future_days)
    
    # Predict future values
    forecast = model.predict(future)

    # Filter predictions for the future period only
    future_forecast = forecast.iloc[-future_days:]
    
    # Save the prediction plot
    plot_filename = f"../predictions/Prophet_{ticker}_Prediction.png"
    fig = model.plot(forecast)
    plt.title(f"Future Prediction for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)  # Free memory
    print(f"Prediction plot saved as {plot_filename}")

    # Save the prediction data to CSV
    csv_filename = f"../predictions/Prophet_{ticker}_Prediction.csv"
    future_forecast.to_csv(csv_filename, index=False)
    print(f"Prediction data saved as {csv_filename}")

    # Save the prediction data to JSON
    json_filename = f"../predictions/Prophet_{ticker}_Prediction.json"
    future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(
        json_filename, orient="records", date_format="iso"
    )
    print(f"Prediction data saved as {json_filename}")

    return future_forecast

# Main function for executing predictions for multiple tickers
def main():
    tickers = ['ADRO.JK', 'BBCA.JK', 'TLKM.JK']
    future_days = 30  # Predict for 30 days ahead
    results = {}

    for ticker in tickers:
        print(f"Making predictions for {ticker}...")
        try:
            forecast = load_and_predict(ticker, future_days)
            results[ticker] = forecast
        except Exception as e:
            print(f"Error in predicting for {ticker}: {e}")

    # Display the last few rows of predictions for each ticker
    for ticker in results:
        print(f"\nLast 5 predictions for {ticker}:")
        print(results[ticker][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

if __name__ == "__main__":
    main()
