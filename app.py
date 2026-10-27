
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈️"
)

st.title("Airline Sentiment Analysis")
st.write(
    "Analyze customer opinions about airlines "
    "using NLP and machine learning."
)

tweet = st.text_area(
    "Enter a tweet",
    placeholder="Example: My flight was delayed again!"
)

if st.button("Analyze Sentiment"):
    if tweet.strip():
        prediction = model.predict(pd.Series([tweet]))[0]

        sentiment_display = {
            "positive": "Positive 😊",
            "negative": "Negative 😞",
            "neutral": "Neutral 😐"
        }

        st.subheader(
            sentiment_display.get(prediction, prediction)
        )
        st.write("Predicted class:", prediction)
    else:
        st.warning("Please enter a tweet.")
