import joblib
import pandas as pd

# 1. Load the model
model = joblib.load('stress_model_rf.pkl')

# 2. Get the feature names the model was trained on
# This ensures we have the correct columns
features = model.feature_names_in_

# 3. Create a dummy data point with the correct number of columns (34)
# We fill it with zeros and then set the most important ones
sample_values = [0] * len(features)
data = pd.DataFrame([sample_values], columns=features)

# Set some realistic values for the top features
data['MEDIAN_RR'] = 630.5
data['HR'] = 75.0

# 4. Get prediction
prediction = model.predict(data)
print(f"Predicted Stress Condition: {prediction[0]}")
