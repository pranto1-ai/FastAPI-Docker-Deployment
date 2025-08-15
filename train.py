import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model directory
os.makedirs('model', exist_ok=True)

# Load data
df = pd.read_csv('heart.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'model/heart_model.joblib')
print("Model saved successfully!")