import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@st.cache_resource
def load_pipeline():
    # Trained pipeline (scaler + logistic regression) saved from the notebook
    return joblib.load("model.pkl")


pipeline = load_pipeline()

st.title("🌾 Crop Recommendation System")
st.write("Enter your soil nutrients and climate conditions to get the most suitable crop.")

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=50.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Recommend Crop", type="primary"):
    X = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
    prediction = pipeline.predict(X)[0]
    st.success(f"Recommended crop: **{prediction}**")

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
        top5 = (
            pd.DataFrame({"Crop": pipeline.classes_, "Confidence (%)": (probs * 100).round(2)})
            .sort_values("Confidence (%)", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        st.subheader("Top 5 predictions")
        st.dataframe(top5, use_container_width=True)
