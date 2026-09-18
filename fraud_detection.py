import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction details and use the predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)
step_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)

if st.button("Predict"):
    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest
    error_balance_orig = oldbalanceOrg - amount - newbalanceOrig
    error_balance_dest = oldbalanceDest + amount - newbalanceDest

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": balance_diff_orig,
        "balanceDiffDest": balance_diff_dest,
        "errorBalanceOrig": error_balance_orig,
        "errorBalanceDest": error_balance_dest,
        "hourOfDay": step_hour
    }])

    expected_cols = set(model.named_steps["prep"].feature_names_in_)
    missing = expected_cols - set(input_data.columns)

    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        prediction = model.predict(input_data)[0]

        st.subheader(f"Prediction: '{int(prediction)}'")

        if prediction == 1:
            st.error("This transaction can be fraud")
        else:
            st.success("This transaction seems not to be fraud")

# use in the terminal -: streamlit run fraud_detection.py