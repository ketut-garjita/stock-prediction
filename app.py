from flask import Flask, request, jsonify
import pickle
import os

# Initialize Flask
app = Flask(__name__)

# Path model
MODEL_DIR = "models"
MODELS = {
    "LSTM_ADRO": os.path.join(MODEL_DIR, "LSTM_ADRO.JK.pkl"),
    "LSTM_BBCA": os.path.join(MODEL_DIR, "LSTM_BBCA.JK.pkl"),
    "LSTM_TLKM": os.path.join(MODEL_DIR, "LSTM_TLKM.JK.pkl"),
    "Prophet_ADRO": os.path.join(MODEL_DIR, "Prophet_ADRO.JK.pkl"),
    "Prophet_BBCA": os.path.join(MODEL_DIR, "Prophet_BBCA.JK.pkl"),
    "Prophet_TLKM": os.path.join(MODEL_DIR, "Prophet_TLKM.JK.pkl"),
    "RandomForest": os.path.join(MODEL_DIR, "RandomForest_stock_models.pkl"),
}

# Load all models when application sterted
loaded_models = {}
for model_name, model_path in MODELS.items():
    with open(model_path, "rb") as f:
        loaded_models[model_name] = pickle.load(f)

# Define a route for the root URL
@app.route('/')
def home():
    return jsonify(message="Stock Prediction API is up and running!")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for prediction. The JSON input must contain:
    - model: The name of the model to be used (e.g., 'LSTM_ADRO', 'Prophet_TLKM')
    - data: Input data for the model (list or dict as per the model requirement)
    """
    data = request.get_json()
    
    # Validate input
    if "model" not in data or "data" not in data:
        return jsonify({"error": "Model name and data are required"}), 400
    
    model_name = data["model"]
    input_data = data["data"]

    # Validate model
    if model_name not in loaded_models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    model = loaded_models[model_name]

    # Perform prediction (fit the model)
    try:
        if model_name.startswith("LSTM") or model_name.startswith("Prophet"):
            prediction = model.predict([input_data])  # Fit the data
        elif model_name == "RandomForest":
            prediction = model.predict(input_data)
        else:
            return jsonify({"error": "Unknown model type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"model": model_name, "prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
