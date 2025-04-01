from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask app
application = Flask(__name__)
app = application

# Secret key for session management
app.secret_key = "your_secret_key"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/prediction', methods=['POST'])
def predict_datapoints():
    try:
        # Extract form data
        gender = request.form.get('gender', '').strip()
        race_ethnicity = request.form.get('race_ethnicity', '').strip()
        parental_level_of_education = request.form.get('parental_level_of_education', '').strip()
        lunch = request.form.get('lunch', '').strip()
        test_preparation_course = request.form.get('test_preparation_course', '').strip()

        # Validate numeric inputs
        try:
            reading_score = float(request.form.get('reading_score', 0))
            writing_score = float(request.form.get('writing_score', 0))
            if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
                flash("Scores must be between 0 and 100.", "error")
                return render_template('index.html')
        except ValueError:
            flash("Invalid input: Scores must be numeric.", "error")
            return render_template('index.html')

        # Ensure required fields are provided
        if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]):
            flash("All fields are required.", "error")
            return render_template('index.html')

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
        
        input_data = data.get_data_as_dataframe()
        logging.info(f"Received Input Data:\n{input_data}")

        # Predict math score
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_data)

        # Validate prediction
        if prediction is None or not isinstance(prediction[0], (int, float)):
            flash("Prediction failed. Please try again.", "error")
            return render_template('index.html')

        # Display prediction
        flash(f"Predicted Math Score: {round(prediction[0], 2)}", "success")
        return render_template('index.html')

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        flash("An unexpected error occurred. Please try again.", "error")
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)  # Change debug=True for development
