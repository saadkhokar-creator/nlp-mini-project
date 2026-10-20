import streamlit as st
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Disaster Tweet Detection",
    page_icon="🚨",
    layout="centered"
)

# ----------------------------
# Download NLTK Resources
# ----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ----------------------------
# Text Cleaning
# ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("About")

st.sidebar.info(
    """
**Disaster Tweet Detection**

Model: Logistic Regression

Features: TF-IDF

Dataset: Kaggle Disaster Tweets

Developer: Your Name
"""
)

# ----------------------------
# Main Page
# ----------------------------
st.title("🚨 Disaster Tweet Detection")

st.write(
    """
Enter a tweet below.

The model will predict whether it describes a **real disaster**
or **not a disaster**.
"""
)

tweet = st.text_area(
    "Enter Tweet",
    height=180,
    placeholder="Example: Massive earthquake destroys hundreds of homes in Japan."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:

        cleaned = clean_text(tweet)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        disaster_probability = probabilities[1]
        not_disaster_probability = probabilities[0]

        st.subheader("Cleaned Tweet")
        st.write(cleaned)

        st.subheader("Prediction")

        if prediction == 1:
            st.error("🚨 Disaster Tweet")
            confidence = disaster_probability
        else:
            st.success("✅ Not a Disaster Tweet")
            confidence = not_disaster_probability

        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(f"**Confidence:** {confidence*100:.2f}%")

        st.subheader("Prediction Probabilities")

        st.write(f"Disaster: **{disaster_probability*100:.2f}%**")

        st.write(f"Not Disaster: **{not_disaster_probability*100:.2f}%**")

# ----------------------------
# Examples
# ----------------------------
st.markdown("---")

st.subheader("Example Tweets")

st.code("Massive earthquake destroys hundreds of homes in Japan.")

st.code("Flash flood warning issued for Mumbai tonight.")

st.code("Going shopping with friends this evening!")

st.code("Happy birthday to my best friend!")
