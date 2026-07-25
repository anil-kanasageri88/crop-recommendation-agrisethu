from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained pipeline (scaler + logistic regression)
# This file is created by your notebook's last step: joblib.dump(pipeline, 'model.pkl')
pipeline = joblib.load("model.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    top5 = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            input_df = pd.DataFrame([values], columns=FEATURES)

            prediction = pipeline.predict(input_df)[0]

            # Show top 5 crop probabilities, since logistic regression
            # gives a confidence score for every crop, not just the top one
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                classes = pipeline.named_steps["model"].classes_
                pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:5]
                top5 = [(name, round(prob * 100, 2)) for name, prob in pairs]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, top5=top5)


if __name__ == "__main__":
    app.run(debug=True)
