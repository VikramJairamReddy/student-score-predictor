from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "student_scores.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_score_model.pkl"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Model Evaluation")
print("----------------")
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# Save the trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully.")