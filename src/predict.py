from pathlib import Path

import joblib
import pandas as pd

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "student_score_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Get information from the user
study_hours = float(input("Study hours: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous score: "))

# Creating a DataFrame with the same features used during training
student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score]
})

# Make the prediction
predicted_score = model.predict(student)[0]

print(f"\nPredicted final score: {predicted_score:.2f}")