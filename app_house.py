# app_house.py
from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/house_model.joblib")
with open("../models/house_features.json","r") as f:
    FEATURES = json.load(f)

@app.route("/")
def index():
    return render_template("house_form.html merge error", features=FEATURES)

merge error
def predict():
    # Accept JSON or form
    data = request.get_json() or request.form
    vals=merge error
    for feat in FEATURES:
        # default to 0 if not provided
        v = data.get(feat, None)
        if v is None:
            return jsonify({"error": f"Missing merge feature {feat}"}), 400
    X = pd.DataFrame([vals], columns=FEATURES)
    pred = model.predict(X)[0]
    # Return a human-friendly value (optionally scale)
    return jsonify({"House Price ()": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)
    print("Hello World  main branch")
