import joblib
import pandas as pd
from flask import Flask, render_template, request

# ---------------------------------
# Initialize Flask App
# ---------------------------------
app = Flask(__name__)

# ---------------------------------
# Load Model & Preprocessor
# ---------------------------------
model = joblib.load("models/stroke_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# ---------------------------------
# Home Page
# ---------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------
# Prediction
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = request.form["gender"]
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = request.form["ever_married"]
        work_type = request.form["work_type"]
        residence_type = request.form["Residence_type"]
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = request.form["smoking_status"]

        # Create DataFrame
        patient_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence_type],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "smoking_status": [smoking_status]
        })

        # Preprocess
        processed_data = preprocessor.transform(patient_data)

        # Prediction
        prediction = model.predict(processed_data)[0]

        # Probability
        probability = model.predict_proba(processed_data)[0][1]

        probability_percent = round(probability * 100, 2)

        # Risk Level
        if probability_percent < 30:
            risk = "🟢 Low Risk"

        elif probability_percent < 60:
            risk = "🟡 Moderate Risk"

        else:
            risk = "🔴 High Risk"

        # Recommendation
        if probability_percent < 30:
            recommendation = "Maintain a healthy lifestyle and continue regular health checkups."

        elif probability_percent < 60:
            recommendation = "Consult a healthcare professional and monitor your health regularly."

        else:
            recommendation = "Please consult a doctor immediately for a detailed medical evaluation."

        return render_template(
            "index.html",
            prediction_text=risk,
            probability=probability_percent,
            recommendation=recommendation
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="Error",
            probability=0,
            recommendation=str(e)
        )


# ---------------------------------
# Run Application
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)