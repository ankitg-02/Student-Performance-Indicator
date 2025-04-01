from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route for Prediction
@app.route('/prediction', methods=['POST'])
def predict_datapoints():
    try:
        # Extract form data with validation
        gender = request.form.get('gender', '').strip()
        race_ethnicity = request.form.get('race_ethnicity', '').strip()
        parental_level_of_education = request.form.get('parental_level_of_education', '').strip()
        lunch = request.form.get('lunch', '').strip()
        test_preparation_course = request.form.get('test_preparation_course', '').strip()

        # Convert scores safely
        try:
            reading_score = float(request.form.get('reading_score', 0))
            writing_score = float(request.form.get('writing_score', 0))
        except ValueError:
            return render_template('error.html', error_message="Invalid input: Scores must be numeric."), 400

        # Ensure required fields are provided
        if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]):
            return render_template('error.html', error_message="All fields are required."), 400

        # Create input data instance
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Convert to DataFrame
        input_data = data.get_data_as_dataframe()
        print("Received Input Data:\n", input_data)

        # Predict math score
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_data)

        # Validate prediction
        if prediction is None or not isinstance(prediction[0], (int, float)):
            return render_template('error.html', error_message="Prediction failed."), 500

        # Show prediction on homepage
        return render_template('index.html', prediction=round(prediction[0], 2))

    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Change debug=False in production
