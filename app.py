
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning project using Random Forest.")

# Load model and dataset
model = joblib.load("fraud_detection_model.pkl")
df = pd.read_csv("creditcard.csv")

st.subheader("📊 Dataset Overview")

total_transactions = len(df)
fraud_transactions = int(df["Class"].sum())
genuine_transactions = total_transactions - fraud_transactions

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Genuine Transactions", f"{genuine_transactions:,}")
col3.metric("Fraud Transactions", f"{fraud_transactions:,}")


st.divider()


st.subheader("🔍 Transaction Analysis")

st.write("Choose an example transaction to test the model.")

col1, col2 = st.columns(2)

if "selected_row" not in st.session_state:
    st.session_state.selected_row = 0

with col1:
    if st.button("✅ Test Genuine Example"):
        st.session_state.selected_row = 92

with col2:
    if st.button("⚠️ Test Fraud Example"):
        st.session_state.selected_row = 541

index = st.number_input(
    "Or select transaction row manually",
    min_value=0,
    max_value=len(df) - 1,
    value=st.session_state.selected_row,
    step=1
)


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


if st.button("Analyze Transaction", type="primary"):
    transaction = df.drop(columns=["Class"]).iloc[[int(index)]]

    prediction = model.predict(transaction)[0]
    probabilities = model.predict_proba(transaction)[0]
    fraud_probability = probabilities[1] * 100

    
    st.session_state.prediction_history.append({
        "Row": int(index),
        "Prediction": "Fraud" if prediction == 1 else "Genuine",
        "Fraud Probability (%)": round(fraud_probability, 2),
        "Actual": "Fraud" if int(df.iloc[int(index)]["Class"]) == 1 else "Genuine"
    })


    st.subheader("Prediction Result")
    st.metric(
        "Estimated Fraud Probability",
        f"{fraud_probability:.2f}%"
    )

    
    st.progress(
        min(max(fraud_probability / 100, 0.0), 1.0)
    )

    if fraud_probability >= 50:
        st.warning("Higher estimated fraud probability")
    else:
        st.info("Lower estimated fraud probability")


    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

    actual_class = int(df.iloc[int(index)]["Class"])

    st.write(
        "Actual transaction type:",
        "Fraud" if actual_class == 1 else "Genuine"
    )

    if prediction == actual_class:
        st.info("Prediction matches the actual label.")
    else:
        st.warning("Prediction does not match the actual label.")



st.divider()
st.caption("Educational demonstration — not for real-world fraud decisions.")



st.subheader("📊 Fraud Distribution (%)")

class_counts = df["Class"].value_counts()

chart_data = pd.DataFrame({
    "Transaction Type": ["Genuine", "Fraud"],
    "Percentage": [
        class_counts.get(0, 0) / len(df) * 100,
        class_counts.get(1, 0) / len(df) * 100
    ]
})

st.bar_chart(
    chart_data,
    x="Transaction Type",
    y="Percentage",
    horizontal=True
)

st.write(f"Genuine: {chart_data.iloc[0]['Percentage']:.2f}%")
st.write(f"Fraud: {chart_data.iloc[1]['Percentage']:.2f}%")


st.divider()
st.subheader("🎯 Random Forest Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Precision", "96.97%")
col2.metric("Recall", "73.33%")
col3.metric("F1-Score", "83.54%")

st.caption(
    "Metrics are based on the model's test-set evaluation."
)



st.divider()
st.subheader("📊 Confusion Matrix")

confusion_data = pd.DataFrame(
    {
        "Predicted Genuine": [56654, 24],
        "Predicted Fraud": [2, 66]
    },
    index=["Actual Genuine", "Actual Fraud"]
)

st.table(confusion_data)

st.caption(
    "Rows = actual transaction type. "
    "Columns = predicted transaction type."
)


st.divider()
st.subheader("📝 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions recorded yet. Analyze a transaction first.")





