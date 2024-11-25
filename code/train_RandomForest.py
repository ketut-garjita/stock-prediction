import yfinance as yf
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix as sklearn_confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
)
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Ticker symbols and dates
TICKER_SYMBOLS = ['ADRO.JK', 'BBCA.JK', 'TLKM.JK']
START_DATE = "2020-01-01"
# START_DATE = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
END_DATE = datetime.today().strftime('%Y-%m-%d')


# ========== Utility Functions ==========
def download_data(tickers, START_DATE, END_DATE):
    """Download historical stock data for given tickers and dates."""
    try:
        return yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker', interval='1d')
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return None


def preprocess_data(data, ticker):
    """Preprocess data for a specific ticker."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]
    else:
        data.columns = data.columns.map(str)    

    # Calculate moving averages, binary target, daily return, and volatility
    data.ffill(inplace=True)
    data[f'{ticker}_MA1'] = data[f'{ticker}_Close'].rolling(window=1).mean()
    data[f'{ticker}_MA30'] = data[f'{ticker}_Close'].rolling(window=30).mean()
    data[f'{ticker}_UpDown'] = ((data[f'{ticker}_Close'] - data[f'{ticker}_Open']) > 0).astype(int)
    data[f'{ticker}_DailyReturn'] = (data[f'{ticker}_Close'] - data[f'{ticker}_Open']) / data[f'{ticker}_Open']
    data[f'{ticker}_Volatility'] = data[f'{ticker}_Close'].pct_change().rolling(window=30).std()
    return data.dropna()


def evaluate_regression_model(y_test, y_pred):
    """Evaluate regression model performance."""
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def log_evaluation_to_json(ticker, target, results, filename='../outputs/RandomForest_Evaluation_Results.json'):
    """Log evaluation results to a JSON file."""
    try:
        with open(filename, 'r') as file:
            evaluations = json.load(file)
    except FileNotFoundError:
        evaluations = {}

    if ticker not in evaluations:
        evaluations[ticker] = {}
    evaluations[ticker][target] = results

    with open(filename, 'w') as file:
        json.dump(evaluations, file, indent=4)


def plot_confusion_matrix(y_test, y_pred, ticker, target, pdf_pages):
    cm = sklearn_confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {ticker} {target}", weight='bold')
    #plt.legend()
    pdf_pages.savefig()  # Save the current figure to PDF
    plt.show(block=False)
    plt.pause(0.5)  # Small pause to allow the plot to render
    plt.close()  # Close the figure


def plot_feature_importances(model, features, ticker, target, pdf_pages):
    if hasattr(model, 'feature_importances_'):     
        feature_importances = model.feature_importances_
        if len(features) != len(feature_importances):
            raise ValueError("Mismatch between features and feature importances length.")
        
        # Create DataFrame for feature importances
        df_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        df_importance = df_importance.sort_values(by='Importance', ascending=False)
            
        # Enable constrained layout
        fig, ax = plt.subplots(constrained_layout=True)

        colors = sns.color_palette("viridis", len(df_importance)) 
        ax = sns.barplot(
            x='Importance',
            y='Feature',
            data=df_importance,
            palette=None  
        )
        
        # Display and Save
        plt.title(f"Feature Importance for {ticker} - {target}", weight='bold')
        
        # Apply colors and annotations to bars
        for bar, color in zip(ax.patches, colors):
            bar.set_facecolor(color)
        
        # Annotate each bar with its value
        for i, bar in enumerate(ax.patches):
            importance_value = bar.get_width()
            ax.text(
                importance_value + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance_value:.2f}",
                color='black',
                ha='left', va='center', fontsize=10
            )
        
        pdf_pages.savefig()  # Save the current figure to PDF
        plt.show(block=False)
        plt.pause(0.5)  # Small pause to allow the plot to render
        plt.close()  # Close the figure
           
    else:
        logging.warning(f"Model for {ticker} - {target} does not have feature importances.")

    
def save_model(model, filename):
    """Save a model to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"\nModel saved to {filename}")


# ========== Core Processing Functions ==========

def tune_hyperparameters(X_train, y_train, model_type='classifier'):
    """Perform hyperparameter tuning using GridSearchCV."""
    if model_type == 'classifier':
        param_grid = {
            'n_estimators': [40, 80],  # Reduce number of trees
            'max_depth': [10, 20],  # Limit tree depth
            'min_samples_split': [2, 5],  # Fewer splits
            'min_samples_leaf': [1, 2],  # Fewer minimum samples
            'max_features': ['sqrt', 'log2']  # Avoid full features
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'regressor':
        param_grid = {
            'n_estimators': [40, 80],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid model type. Use 'classifier' or 'regressor'.")

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=7,  # 7-fold cross-validation (default=5)
        scoring='accuracy' if model_type == 'classifier' else 'r2',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_
    

def train_and_evaluate_with_tuning(data, ticker, target, model_type, pdf_pages):
    feature_map = {
        'UpDown': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_MA1'],
        'DailyReturn': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_High', f'{ticker}_Low'],
        'Volatility': [f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_Volume', f'{ticker}_MA30']
    }

    target_suffix = target.replace(f'{ticker}_', '')
    features = feature_map.get(target_suffix)

    if not features:
        raise ValueError(f"Unknown target: {target}")

    X = data[features]
    y = data[f'{ticker}_{target_suffix}']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Perform hyperparameter tuning
    best_model, best_params = tune_hyperparameters(X_train, y_train, model_type)

    # Evaluate tuned model
    y_pred = best_model.predict(X_test)

    if model_type == 'classifier':
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        plot_confusion_matrix(y_test, y_pred, ticker, target, pdf_pages)
        logging.info(f"Tuned Model Accuracy for {ticker} {target}: {accuracy}")
        logging.info(f"Best Parameters: {best_params}")
        return best_model, accuracy, features
    else:
        metrics = evaluate_regression_model(y_test, y_pred)
        plot_residuals(y_test, y_pred, target, pdf_pages)
        log_evaluation_to_json(ticker, target, metrics)
        logging.info(f"Tuned Model Metrics for {ticker} {target}: {metrics}")
        logging.info(f"Best Parameters: {best_params}")
        return best_model, metrics['R2'], features


def plot_residuals(y_true, y_pred, target_name, pdf_pages):
    residuals = y_true - y_pred  # Calculate residuals
    # Display
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color="blue", line_kws={'color': 'red', 'lw': 1})
    plt.title(f'Residual Plot for {target_name}', fontsize=16, weight='bold')
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.axhline(0, color='black', lw=2)  # Horizontal line at y=0
    # plt.legend()
    pdf_pages.savefig()  # Save the current figure to PDF
    plt.show(block=False)
    plt.pause(0.5)  # Small pause to allow the plot to render
    plt.close()  # Close the figure
    

def save_final_results_to_json(results, filename='final_results.json'):
    """Save final results dictionary to a JSON file."""
    try:
        with open(f"../outputs/{filename}", 'w') as file:
            json.dump(results, file, indent=4)
        logging.info(f"Final results saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving final results: {e}")
       
    # Group results by category
    categories = ["UpDown", "DailyReturn", "Volatility"]
    grouped_results = {category: {} for category in categories}
    
    for key, value in results.items():
        for category in categories:
            if category in key:
                grouped_results[category][key] = value
    
    # Print grouped results
    for category, items in grouped_results.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    # Optionally save grouped results as JSON
    with open("../outputs/RandomForest_Grouped_Results_Report.json", "w") as f:
        json.dump(grouped_results, f, indent=4)


# ========== Main Program ==========
def main():
    start_time = datetime.now()
    logging.info("Start downloading data, creating, training, evaluating, and tuning models...")
    data = download_data(TICKER_SYMBOLS, START_DATE, END_DATE)

    if data is None:
        print("Data download failed. Exiting program.")
        return
    
    # Open a PDF file to save all plots
    with PdfPages('../outputs/RandomForest_Stock_Analysis_Plots.pdf') as pdf_pages:
        models = {}  # Dictionary for storing models
        results = {}  # Dictionary for storing scores
        
        for ticker in TICKER_SYMBOLS:
            print(f"Processing ticker: {ticker}")
            ticker_data = preprocess_data(data, ticker)
            
            for target, model_type in [
                (f'{ticker}_UpDown', 'classifier'),
                (f'{ticker}_DailyReturn', 'regressor'),
                (f'{ticker}_Volatility', 'regressor')
            ]:
                # Use tuned training and evaluation function
                model, score, selected_features = train_and_evaluate_with_tuning(
                    ticker_data, ticker, target, model_type, pdf_pages
                )
                plot_feature_importances(model, selected_features, ticker, target, pdf_pages)
                
                # Store models and scores
                models[target] = model  # Save trained models
                results[target] = score  # Save performance scores
        
        # Save all models and scores into a single pickle file
        all_results = {'models': models, 'scores': results}
        save_model(all_results, '../models/RandomForest_stock_models.pkl')
        logging.info("\nProcessing completed.")
        
        # Save the results to a JSON file for report
        save_final_results_to_json(results, 'RadnomForest_Final_Results.json')
  
    end_time = datetime.now()
    time_difference = end_time - start_time
    duration = str(time_difference).split('.')[0] 
    print("\nElapsed time (HH:MM:SS):", duration)
    print("All plots saved to 'stock_analysis_plots.pdf'.")
    

if __name__ == '__main__':
    main()
