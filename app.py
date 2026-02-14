import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="centered")

# Title
st.markdown("## 🧠 Parkinson’s Disease Detection")
st.markdown("Interactive demo using XGBoost. Use sliders, presets, or upload CSV for batch predictions.")

# -------------------------
# Feature Groups & Ranges
# -------------------------
feature_groups = {
    "Frequency Features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
    "Jitter Features": ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"],
    "Shimmer Features": ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"],
    "Other Features": ["NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
}

feature_ranges = {
    "MDVP:Fo(Hz)": (100,300,0.1), "MDVP:Fhi(Hz)": (200,400,0.1), "MDVP:Flo(Hz)": (50,200,0.1),
    "MDVP:Jitter(%)": (0.0005,0.020,0.0001), "MDVP:Jitter(Abs)": (0.000002,0.005,0.000001),
    "MDVP:RAP": (0.0005,0.020,0.0001), "MDVP:PPQ": (0.0005,0.020,0.0001), "Jitter:DDP": (0.0005,0.020,0.0001),
    "MDVP:Shimmer": (0.005,0.150,0.001), "MDVP:Shimmer(dB)": (0.02,2.0,0.01),
    "Shimmer:APQ3": (0.005,0.150,0.001), "Shimmer:APQ5": (0.005,0.150,0.001),
    "MDVP:APQ": (0.005,0.150,0.001), "Shimmer:DDA": (0.005,0.200,0.001),
    "NHR": (0.0003,0.749,0.001), "HNR": (1,40,0.1),
    "RPDE": (0.15,0.99,0.01), "DFA": (0.50,0.90,0.01),
    "spread1": (-7,-1,0.01), "spread2": (0.40,3.0,0.01),
    "D2": (1,6,0.01), "PPE": (0.02,1.0,0.01)
}

# -------------------------
# Presets
# -------------------------
st.sidebar.header("Presets")
preset = st.sidebar.selectbox("Choose a preset:", ("None", "Typical Healthy", "Typical Parkinson's"))
preset_factor = None
if preset == "Typical Healthy": preset_factor = 0.2
if preset == "Typical Parkinson's": preset_factor = 0.8

# -------------------------
# Sliders
# -------------------------
st.markdown("### Voice Feature Inputs")
feature_values = {}

for group, feats in feature_groups.items():
    with st.expander(group):
        for f in feats:
            low, high, step = feature_ranges[f]
            default_val = low + 0.5*(high-low)
            if preset_factor is not None:
                default_val = low + preset_factor*(high-low)
            feature_values[f] = st.slider(f, float(low), float(high), float(default_val), step=float(step))

input_array = np.array(list(feature_values.values())).reshape(1,-1)

# -------------------------
# Batch CSV Upload
# -------------------------
st.markdown("### Upload CSV (optional)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
batch_df = None
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Preview:")
    st.dataframe(batch_df.head())

# -------------------------
# Predict
# -------------------------
if st.button("Predict"):

    # Load dataset or dummy
    try:
        df = pd.read_csv("parkinsons_data.csv")
        X_train = df.drop(columns=["status"]).values
        y_train = df["status"].values
    except:
        np.random.seed(42)
        X_train = np.random.rand(50,22)
        y_train = np.random.randint(0,2,50)

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False, random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Single prediction
    pred_prob = pipeline.predict_proba(input_array)[0][1]
    pred = pipeline.predict(input_array)[0]

    # -------------------------
    # Gradient Probability Bar + Percentage
    # -------------------------
    if pred==1:
        color_start = "#ff9999"
        color_end = "#ff3333"
        pred_text = "Prediction: Parkinson’s Disease Detected"
    else:
        color_start = "#99ff99"
        color_end = "#33cc33"
        pred_text = "Prediction: Healthy (No Parkinson’s)"

    st.markdown(f"### {pred_text}")
    st.markdown(f"""
    <div style="background-color:#eee; padding:5px; border-radius:5px;">
        <div style="
            width:{pred_prob*100}%;
            height:25px;
            background: linear-gradient(to right, {color_start}, {color_end});
            border-radius:5px;
            transition: width 0.5s;">
        </div>
    </div>
    <p style="margin-top:5px; font-weight:bold;">Probability: {pred_prob*100:.2f}%</p>
    """, unsafe_allow_html=True)

    # -------------------------
    # Feature Importance
    # -------------------------
    st.markdown("### Feature Importance")
    model = pipeline.named_steps['model']
    importances = model.feature_importances_
    feat_names = list(feature_values.keys())
    importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,5))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="#444444")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    st.pyplot(plt)

    # -------------------------
    # Batch prediction
    # -------------------------
    if batch_df is not None:
        X_batch = batch_df.values
        probs = pipeline.predict_proba(X_batch)[:,1]
        preds = pipeline.predict(X_batch)
        batch_df["Prediction"] = ["Parkinson" if p==1 else "Healthy" for p in preds]
        batch_df["Probability"] = probs
        st.markdown("---")
        st.write("### Batch Prediction Results")
        st.dataframe(batch_df)
