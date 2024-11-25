import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

tickers = ['ADRO.JK', 'BBCA.JK', 'TLKM.JK']
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

start_time = datetime.now()

# Download yfinance stock data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', interval='1d')

# Align MultiIndex columns to single-level columns
data.columns = ['_'.join(col).strip() for col in data.columns]

# Select the Close price for each ticker
close_prices = data.filter(like='_Close')

# Train a model for each stock
future_periods = 30  # 

# Prophet model training function
def train_and_forecast(df, ticker, future_periods):
    # Data format for Prophet
    
    df_prophet = df.reset_index()[['Date', f'{ticker}_Close']]
    df_prophet.columns = ['ds', 'y'] 

    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None) # remove timezone

    # Create Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    # Save the model
    model_filename = f"../models/Prophet_{ticker}.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"\nModel for {ticker} saved as {model_filename}\n")

    # Create a dataframe for a future period
    future = model.make_future_dataframe(periods=future_periods)

    # Prediction
    forecast = model.predict(future)

    # Plot result
    fig = model.plot(forecast)
    plt.title(f"Forecast for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show(block=False)
    plt.pause(0.5)  # Small pause to allow the plot to render

    # Plot result
    fig = model.plot(forecast)
    plt.title(f"Forecast for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plot_filename = f"../outputs/Prophet_{ticker}_Forecast.png"
    plt.savefig(plot_filename)
    print(f"Plot for {ticker} saved as {plot_filename}")
    plt.close(fig)  # Close the plot to free memory

    # Save the forecast data to CSV
    forecast_filename = f"../outputs/Prophet_{ticker}_Forecast.csv"
    forecast.to_csv(forecast_filename, index=False)
    print(f"Forecast data for {ticker} saved as {forecast_filename}")

    return forecast

# Execute model for each ticker
results = {}
for ticker in tickers:
    print(f"Training model for {ticker}...")
    forecast = train_and_forecast(close_prices, ticker, future_periods)
    results[ticker] = forecast

# Show prediction result
for ticker in results:
    print(f"\nForecast for {ticker}:")
    print(results[ticker][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

print("")


# Comparison of Actual vs Predicted Stock Prices (IDR) 

for ticker in tickers:
    # Select Close prices for the ticker
    close_prices = data.filter(like=f'{ticker}_Close')
    close_prices = close_prices.reset_index()[['Date', f'{ticker}_Close']]
    close_prices.columns = ['ds', 'y']  # Rename columns for Prophet

    # Remove timezone information from dates
    close_prices['ds'] = pd.to_datetime(close_prices['ds']).dt.tz_localize(None)

    # Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(close_prices)

    # Create future dataframe
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)

    # Merge actual and predicted values
    df_compare = pd.merge(close_prices, forecast[['ds', 'yhat']], on='ds', how='outer')

    # Display and save
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare['ds'], df_compare['y'], label='Actual', color='blue')
    plt.plot(df_compare['ds'], df_compare['yhat'], label='Predicted', color='red')
    plt.title(f'Comparison: Actual vs Predicted Stock Prices (IDR) for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (IDR)')
    plt.legend()
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../outputs/Prophet_{ticker}_Comparison.png")
    plt.show(block=False)
    plt.pause(0.5)  # Small pause to allow the plot to render
   
    # Save the combined data to forecast_recent
    forecast_recent = df_compare.copy()

    # Show process duration
    end_time = datetime.now()
    duration = str(end_time - start_time).split('.')[0]
    print("\nElapsed time (HH:MM:SS):", duration)