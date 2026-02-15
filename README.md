
#   Parkinsons Disease Detection web app

1.This repository contains an **interactive machine learning web application** that predicts the presence of **Parkinson’s Disease** based on input voice and biomedical features.  

2.The app is built using **Streamlit** and utilizes an **XGBoost classifier** for reliable prediction.

## Project Overview

1.Parkinson’s Disease is a progressive neurological disorder that affects movement and vocal signals due to a reduction in dopamine-producing neurons. Early detection is crucial in improving patient care and treatment planning.

2.This project uses machine learning to analyze biomedical voice measurements and other health-related features to predict whether a user is likely to have Parkinson’s Disease.

# Features

# Intelligent Prediction
- Uses an **XGBoost machine learning model** for classifying healthy vs Parkinson’s samples.
- Integrated through a **Scikit-Learn Pipeline** with **StandardScaler** for proper feature scaling.

### Interactive User Input
- Users can enter values through **slider controls** for 22 different features.
- Features grouped logically for easy navigation (e.g., “Frequency Features”, “Jitter”, “Shimmer”).

### Preset Profiles
- Two quick presets:
  - **Typical Healthy**
  - **Typical Parkinson’s**
- Helps users insightfully test app behavior without entering all values manually.

# Batch Prediction Support
- Upload a **CSV file** of data for batch predictions.
- The app will display predicted class and probability for each sample.

## 🛠️ Technologies Used

- **Streamlit** — for web interface
- **XGBoost (XGBClassifier)** — machine learning model
- **Scikit-Learn Pipeline & StandardScaler** — preprocessing and modeling
- **NumPy & Pandas** — data handling
- **Matplotlib** — plots and charts

## How It Works

1. User inputs feature values via sliders.
2. The app forms the feature vector for prediction.
3. The model predicts class (Healthy or Parkinson’s).
4. Probability and supportive visuals are displayed.
## Live Demo (Deployed)

**Click here to try the live app:**  
https://parkinsonsdiseasedetection.streamlit.app
