import streamlit as st
import joblib
import numpy as np
import math
import uuid
import datetime

# Load model
model = joblib.load("logistic_model.joblib")

# Placeholder values — replace with actual training stats
amount_mean = 1500.0
amount_std = 1200.0
large_amount_threshold = 5000
distance_threshold = 100

# Risky devices — replace with actual list from training
risky_devices = ["POS", "Unknown", "CustomDevice"]

# --- Streamlit UI ---
st.set_page_config(page_title="Fraud Detection", page_icon="🕵️", layout="centered")
st.title("🕵️‍♂️ Transaction Fraud Detector")
st.markdown("Fill in transaction details to check for fraud risk.")


#---- Fake inp----

st.subheader("📋 Transaction Metadata")
transaction_id = st.text_input("Transaction ID", value=str(uuid.uuid4()))
customer_id = st.text_input("Customer ID", value=f"CUST{np.random.randint(1000,9999)}")
card_number = st.text_input("Card Number (masked)", value="XXXX-XXXX-XXXX-1234")
timestamp = st.date_input("Transaction Date", value=datetime.date.today())
device_fingerprint = st.text_input("Device Fingerprint", value=str(uuid.uuid4())[:8])
ip_address = st.text_input("IP Address", value="192.168.0.1")

# --- Inputs ---
merchant_category = st.selectbox("Merchant Category", ["Retail", "Food", "Travel", "Electronics", "Other"])
merchant_type = st.selectbox("Merchant Type", ["Online", "In-store", "Subscription", "ATM"])
amount = st.number_input("Transaction Amount", min_value=0.0, max_value=100000.0, value=500.0)
currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"])
country = st.selectbox("Country", ["India", "USA", "UK", "Germany", "Other"])
card_type = st.selectbox("Card Type", ["Credit", "Debit", "Prepaid"])
card_present = st.radio("Card Present?", ["Yes", "No"])
device = st.selectbox("Device Used", ["Mobile", "Desktop", "POS", "Tablet", "Unknown"])
channel = st.selectbox("Transaction Channel", ["App", "Web", "ATM", "POS"])
distance_from_home = st.slider("Distance from Home (km)", 0, 500, 10)
hour = st.slider("Transaction Hour (0–23)", 0, 23, 14)

# --- Feature Engineering ---
is_night = 1 if hour < 6 or hour >= 20 else 0
is_peak_hour = 1 if 8 <= hour <= 18 else 0

def hour_bin(h):
    if 6 <= h < 12: return "morning"
    elif 12 <= h < 17: return "afternoon"
    elif 17 <= h < 20: return "evening"
    else: return "night"

hour_bin_val = hour_bin(hour)
hour_sin = math.sin(2 * math.pi * hour / 24)
hour_cos = math.cos(2 * math.pi * hour / 24)

is_large_amount = amount > large_amount_threshold
log_amount = np.log1p(amount)
amount_zscore = (amount - amount_mean) / amount_std

is_remote = distance_from_home > distance_threshold
is_card_not_present = 1 if card_present == "No" else 0
device_risk_score = 1 if device in risky_devices else 0
channel_device_combo = f"{channel}_{device}"

# --- Encoding (simplified hash-based) ---
def encode(val):
    return hash(val) % 1000 if isinstance(val, str) else int(val)

# --- Final Feature Vector (23 features) ---
features = [
    encode(merchant_category),
    encode(merchant_type),
    amount,
    encode(currency),
    encode(country),
    encode(card_type),
    1 if card_present == "Yes" else 0,
    encode(device),
    encode(channel),
    distance_from_home,
    hour,
    is_night,
    is_peak_hour,
    encode(hour_bin_val),
    hour_sin,
    hour_cos,
    int(is_large_amount),
    log_amount,
    amount_zscore,
    int(is_remote),
    is_card_not_present,
    device_risk_score,
    encode(channel_device_combo)
]

input_array = np.array(features).reshape(1, -1)

# --- Prediction ---
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! Confidence: {probability:.2f}")
    else:
        st.success(f"✅ Transaction is Safe. Confidence: {1 - probability:.2f}")

    st.markdown("📊 **Model Confidence Level:**")
    st.progress(probability if prediction == 1 else 1 - probability)
