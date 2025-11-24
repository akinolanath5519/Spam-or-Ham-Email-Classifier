import streamlit as st
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Clean function
def simple_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Load model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("📩 Spam Message Classifier")

msg = st.text_area("Enter a message")

if st.button("Predict"):
    cleaned = simple_clean(msg)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]

    output = "🚨 SPAM" if pred == 1 else "✔️ NOT SPAM"
    st.subheader(output)
