from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import pickle


# === CONFIG ===
DATA_PATH = "Dataset/data.csv"
MODEL_PATH = "model/model.pkl"

app = Flask(__name__)

# === STEP 1: Train model ===
def train_model():
    print("🔁 Training model...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at '{DATA_PATH}'")

    df = pd.read_csv(DATA_PATH, encoding='latin1')
    df = df[['so2', 'no2', 'pm2_5']].dropna()

    # Create a dummy AQI for now (you should replace this with real AQI values if available)
    df['AQI'] = df['so2'] * 0.25 + df['no2'] * 0.25 + df['pm2_5'] * 0.5

    features = ['so2', 'no2', 'pm2_5']
    target = 'AQI'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"✅ Model trained. R² score: {r2_score(y_test, y_pred):.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

# === STEP 2: Load or train model ===
if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)

# === STEP 3: Flask routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            so2 = float(request.form['so2'])
            no2 = float(request.form['no2'])
            pm2_5 = float(request.form['pm2_5'])

            features = np.array([[so2, no2, pm2_5]])
            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = "Invalid input"
            print("⚠️ Prediction error:", e)

    return render_template("index.html", prediction=prediction)

# === STEP 4: Run app ===

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

