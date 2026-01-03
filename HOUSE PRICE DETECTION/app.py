import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Sidebar
st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("### Enter House Details")

area = st.sidebar.number_input("Area (sq ft)", min_value=300, max_value=10000, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)

predict_btn = st.sidebar.button("🔮 Predict Price")


# Main Title
st.title("🏡 House Price Prediction System")
st.write("This application predicts house prices using **Machine Learning Regression Model**.")

# Prediction Section
if predict_btn:
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated House Price: ₹ {int(prediction[0]):,}")

# Divider
st.markdown("---")

# Load dataset 
df = pd.read_csv("House Price Prediction Dataset.csv")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("### Area vs Price")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Area", y="Price", data=df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("### Bedrooms vs Price")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Bedrooms", y="Price", data=df, ax=ax2)
    st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👨‍💻 **Developed by:** Neelakshi Chaturvedi")
st.markdown("📌 **Project Type:** Internship / Data Science Project")