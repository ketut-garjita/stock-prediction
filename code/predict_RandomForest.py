import pickle
import pandas as pd
import yfinance as yf
import logging
import json
from datetime import datetime, timedelta

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Ticker symbols
TICKER_SYMBOLS = ['ADRO.JK', 'BBCA.JK', 'TLKM.JK']
START_DATE = "2020-01-01"
# START_DATE = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
# END_DATE = datetime.today().strftime('%Y-%m-%d')
END_DATE = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')

def download_data(tickers, start_date, end_date):
    """Download historical stock data for given tickers and dates."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', interval='1d')
        return data
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return None


def preprocess_data(data, ticker):
    """Preprocess data for a specific ticker."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]
    else:
        data.columns = data.columns.map(str)

    # Calculate moving averages, daily return, and volatility
    data[f'{ticker}_MA1'] = data[f'{ticker}_Close'].rolling(window=1).mean()
    data[f'{ticker}_MA30'] = data[f'{ticker}_Close'].rolling(window=30).mean()
    data[f'{ticker}_Volatility'] = data[f'{ticker}_Close'].pct_change().rolling(window=30).std()

    return data.dropna()


def load_model(filename):
    """Load model from pickle file."""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def make_prediction(data, model, features):
    """Make predictions using the loaded model."""
    try:
        X = data[features]
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None


def main():
    # Load models and results
    model_data = load_model('../models/RandomForest_stock_models.pkl')
    if not model_data:
        print("Failed to load model data. Exiting.")
        return

    models = model_data['models']
    results = model_data['scores']
    print(f"Loaded models for: {list(models.keys())}")

    # Download recent data
    data = download_data(TICKER_SYMBOLS, START_DATE, END_DATE)
    if data is None:
        print("Data download failed. Exiting program.")
        return

    predictions = {}
    for ticker in TICKER_SYMBOLS:
        logging.info(f"Processing ticker: {ticker}")
        ticker_data = preprocess_data(data, ticker)
        ticker_data.reset_index(inplace=True)  # Ensure 'Date' is a column

        for target, model in models.items():
            if ticker not in target:
                continue  # Skip models not associated with this ticker

            # Identify the feature set based on the target
            feature_map = {
                'UpDown': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_MA1'],
                'DailyReturn': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_High', f'{ticker}_Low'],
                'Volatility': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_Volume', f'{ticker}_MA30']
            }

            target_suffix = target.replace(f'{ticker}_', '')
            features = feature_map.get(target_suffix)

            if features is None:
                logging.warning(f"No features mapped for {target}")
                continue

            # Predict
            logging.info(f"Predicting {target}...")
            ticker_predictions = make_prediction(ticker_data, model, features)

            if ticker_predictions is not None:
                predictions[target] = pd.DataFrame({
                    'Date': ticker_data['Date'].dt.strftime('%Y-%m-%d'),  # Format Date
                    'Prediction': ticker_predictions
                }).to_dict(orient='records')
    
    # Save predictions to JSON
    try:
        with open('../predictions/RandomForest_Predictions.json', 'w') as file:
            json.dump(predictions, file, indent=4, default=str)  # Use default=str for Date serialization
        logging.info("Predictions saved to RandomForest_predictions.json'")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")


if __name__ == '__main__':
    main()
