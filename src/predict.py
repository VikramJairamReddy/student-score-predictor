from pathlib import Path

import joblib
import pandas as pd

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "student_score_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Get information from the user
while True:
    try:
        study_hours = float(input("Study hours (0–24): "))
        if 0 <= study_hours <= 24:
            break
        print("Study hours must be between 0 and 24.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        attendance = float(input("Attendance percentage (0–100): "))
        if 0 <= attendance <= 100:
            break
        print("Attendance must be between 0 and 100.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        previous_score = float(input("Previous score (0–100): "))
        if 0 <= previous_score <= 100:
            break
        print("Previous score must be between 0 and 100.")
    except ValueError:
        print("Please enter a valid number.")

# Creating a DataFrame with the same features used during training
student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score]
})

# Make the prediction
predicted_score = model.predict(student)[0]
predicted_score = max(0, min(100, predicted_score))

print(f"\nPredicted final score: {predicted_score:.2f}")