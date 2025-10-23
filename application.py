from flask import Flask, render_template, request
import numpy as np
import joblib
from config.paths_config import MODEL_PATH, SCALER_PATH

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route('/')
def home():
    return render_template("index.html", predictions = None)


@app.route('/predict', methods = ["POST"])
def predict():
    try:
        healthcare_cost = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        treatment_type = int(request.form["treatment_type"])
        diabetes = int(request.form["diabetes"])
        mortality_rate = float(request.form["mortality_rate"])


        input = np.array([[healthcare_cost, tumor_size, treatment_type, diabetes, mortality_rate]])

        scaled_input = scaler.transform(input)

        prediction = model.predict(scaled_input)[0]

        return render_template("index.html", prediction = prediction)

    except Exception as e:
        return str(e)
    

if __name__ == "__main__":

    app.run(debug = True, host = "0.0.0.0", port = 5000)