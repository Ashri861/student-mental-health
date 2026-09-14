from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    risk_level = None

    if request.method == "POST":
        raw = {
            "Gender": request.form["Gender"],
            "Age": float(request.form["Age"]),
            "Academic Pressure": float(request.form["Academic Pressure"]),
            "Work Pressure": float(request.form["Work Pressure"]),
            "CGPA": float(request.form["CGPA"]),
            "Study Satisfaction": float(request.form["Study Satisfaction"]),
            "Job Satisfaction": float(request.form["Job Satisfaction"]),
            "Sleep Duration": request.form["Sleep Duration"],
            "Dietary Habits": request.form["Dietary Habits"],
            "Degree": request.form["Degree"],
            "Have you ever had suicidal thoughts ?": request.form["Suicidal Thoughts"],
            "Work/Study Hours": float(request.form["Work/Study Hours"]),
            "Financial Stress": float(request.form["Financial Stress"]),
            "Family History of Mental Illness": request.form["Family History"],
        }

        categorical_cols = [
            "Gender", "Sleep Duration", "Dietary Habits",
            "Degree", "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness"
        ]

        for col in categorical_cols:
            raw[col] = label_encoders[col].transform([raw[col]])[0]

        input_array = np.array([[raw[col] for col in feature_columns]])
        input_scaled = scaler.transform(input_array)

        result = model.predict(input_scaled)[0]

        if result == 1:
            prediction = "At Risk of Depression"
            risk_level = "high"
        else:
            prediction = "Not At Risk"
            risk_level = "low"

    return render_template("index.html", prediction=prediction, risk_level=risk_level)

if __name__ == "__main__":
    app.run(debug=True)