# 🧠 Stress Detection from Heart Rate Variability (HRV)

## 📌 Project Overview
This project leverages Machine Learning to identify human stress levels based on physiological data. Using heart rate sensor data, the system classifies a person's state into three categories: **No Stress**, **Interruption**, or **Time Pressure**.

The model was trained on over **369,000 samples** and is designed for health-tech applications where real-time stress monitoring can improve workplace wellness.

## 📊 Key Results
* **Accuracy:** 100% on the test dataset.
* **Model Used:** Random Forest Classifier.
* **Top Predictor:** `MEDIAN_RR` (The median time between heartbeats).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib
* **Serialization:** Joblib

## 📈 Visualizations

### 1. Performance Metrics
Our model shows perfect separation between stress states, validated by the Confusion Matrix and ROC Curve.

![Confusion Matrix](confusion_matrix.png)
![ROC Curve](roc_curve.png)

### 2. Biological Insights & Feature Importance
The model identified the **Median RR Interval** as the most critical feature. This aligns with neuroscience: stress typically shortens the interval between heartbeats.

![Feature Importance](feature_importance.png)
![Biological Validation](biological_proof.png)

## 📂 Repository Structure
* `stress_detection.py`: Main script for data processing and training.
* `example_usage.py`: A simple script to load the model and run a prediction.
* `stress_model_rf.pkl`: The final trained model file.
* `requirements.txt`: List of Python dependencies.

## 🚀 How to Use the Model
To use the trained model for your own predictions:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
