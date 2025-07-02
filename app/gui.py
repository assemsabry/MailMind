import streamlit as st
from src.predict_with_bert import predict_with_roberta

st.set_page_config(page_title="MailMind", layout="centered")
st.title("\U0001F4EC MailMind - Smart Email Classifier (RoBERTa Powered)")

email_input = st.text_area("Paste your email content here:")

if st.button("Classify"):
    if email_input.strip():
        result = predict_with_roberta(email_input)
        st.success(f"Predicted Category: **{result}**")
    else:
        st.warning("Please enter some email text.")
