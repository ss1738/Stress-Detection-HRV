import joblib
import pandas as pd

# Load the model
model = joblib.load('stress_model_rf.pkl')

# Create a sample data point (ensure all 34 features are present)
features = model.feature_names_in_
sample_data = pd.DataFrame([[0.0] * len(features)], columns=features)

# Set specific values for important features
sample_data['MEDIAN_RR'] = 630.5
sample_data['HR'] = 75.0

prediction = model.predict(sample_data)
print(f"Predicted Stress Condition: {prediction[0]}")
