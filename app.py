# app.py

import streamlit as st
import pickle
import re
import string

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# UI
st.title("📧 Spam Message Detector")
st.write("Type a message and check if it is SPAM or NOT")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == "spam":
        st.error("🚨 This message is SPAM!")
    else:
        st.success("✅ This message is NOT spam")
