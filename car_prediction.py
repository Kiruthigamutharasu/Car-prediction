# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import io

# -------------------------------
# Step 1: Login Page
# -------------------------------
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------
# Step 2: Load Model & Data
# -------------------------------
st.set_page_config(page_title="Car Price Dashboard", layout="wide")
st.title("🚗 Car Price Prediction Dashboard")

model, X_test, y_test = pickle.load(open("random_forest_model.pkl", "rb"))
df = pd.read_csv("car_dataset.csv")

# -------------------------------
# Step 3: Sidebar Inputs
# -------------------------------
st.sidebar.header("📥 Enter Car Details")

present_price = st.sidebar.slider("Present Price (in lakhs)", 0.0, 100.0, 5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=1)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", [0, 1, 2, 3])
car_age = st.sidebar.slider("Car Age", 0, 20, 5)

# -------------------------------
# Step 4: Predict Button
# -------------------------------
if st.sidebar.button("🔍 Predict"):
    input_df = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'Car_Age': [car_age]
    })

    input_df['Price_per_KM'] = input_df['Present_Price'] / input_df['Kms_Driven']
    input_df['Age_Category'] = pd.cut(input_df['Car_Age'], bins=[0,3,7,100], labels=['New','Mid','Old'])
    le = LabelEncoder()
    input_df['Age_Category'] = le.fit_transform(input_df['Age_Category'])

    input_df = pd.get_dummies(input_df, columns=['Fuel_Type', 'Transmission', 'Seller_Type'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    predicted_price = model.predict(input_df)[0]
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Price", f"₹{predicted_price:.2f} Lakhs")
    col2.metric("📊 Model Accuracy", f"{r2*100:.2f}%")
    col3.metric("📉 MAE", f"₹{mae*100000:.0f}")

    st.subheader("📉 Predicted vs Present Price")

    # Predict on full dataset
    X_full = X_test.copy()
    y_full_pred = model.predict(X_full)

    # Get corresponding Present_Price column from original df
    present_prices = df.loc[X_test.index, 'Present_Price']

     # Plot
    fig3, ax3 = plt.subplots()
    ax3.scatter(present_prices, y_full_pred, alpha=0.7, color='teal')
    ax3.set_xlabel("Present Price (Lakhs)")
    ax3.set_ylabel("Predicted Selling Price (Lakhs)")
    ax3.set_title("Predicted vs Present Price")
    st.pyplot(fig3)


    st.subheader("📊 Price Comparison")
    fig, ax = plt.subplots()
    sns.histplot(df['Selling_Price'], bins=20, ax=ax, label="Dataset")
    ax.axvline(predicted_price, color='red', linestyle='--', label="Your Prediction")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -------------------------------
# Step 5: Batch Prediction
# -------------------------------
st.markdown("---")
st.subheader("📁 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    # Feature engineering
    batch_df['Price_per_KM'] = batch_df['Present_Price'] / batch_df['Kms_Driven']
    batch_df['Age_Category'] = pd.cut(batch_df['Car_Age'], bins=[0,3,7,100], labels=['New','Mid','Old'])
    le = LabelEncoder()
    batch_df['Age_Category'] = le.fit_transform(batch_df['Age_Category'])

    batch_df = pd.get_dummies(batch_df, columns=['Fuel_Type', 'Transmission', 'Seller_Type'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in batch_df.columns:
            batch_df[col] = 0
    batch_df = batch_df[model.feature_names_in_]

    batch_predictions = model.predict(batch_df)
    result_df = pd.DataFrame(batch_predictions, columns=["Predicted_Price"])
    st.write(result_df)

    # -------------------------------
    # Step 6: Download Prediction Report
    # -------------------------------
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Prediction Report", data=csv_buffer.getvalue(), file_name="predicted_prices.csv", mime="text/csv")

# -------------------------------
# Step 7: Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Kiruthiga ")
