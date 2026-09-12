from pathlib import Path

import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_score_model.pkl"

# Load the real dataset
df = pd.read_csv(DATA_PATH, sep=";")

# Features and target
features = ["studytime", "absences", "failures"]

X = df[features]
y = df["G3"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5 # Calculate root mean squared error
r2 = r2_score(y_test, predictions)

print("Real Student Performance Model")
print("--------------------------------")
print("Features:", features)
print("Target: G3 (final grade)")
print()
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# Show learned coefficients
print("\nModel coefficients")
print("------------------")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

# Save the trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully.")
print(MODEL_PATH)