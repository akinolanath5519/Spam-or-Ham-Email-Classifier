# import streamlit as st
# import nltk
# import re
# import pickle
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # # Download to a folder in your project
# # nltk.download("punkt", download_dir="./nltk_data")
# # nltk.download("stopwords", download_dir="./nltk_data")
# # nltk.download("wordnet", download_dir="./nltk_data")

# # # Tell NLTK to use this folder
# # nltk.data.path.append("./nltk_data")

# # Use preloaded folder
# nltk.data.path.append("./nltk_data")

# # Clean function
# def simple_clean(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     tokens = nltk.word_tokenize(text)
#     stop_words = set(stopwords.words("english"))
#     tokens = [w for w in tokens if w not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens]
#     return " ".join(tokens)

# # Load model
# model = pickle.load(open("spam_model.pkl", "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# st.title("📩 Spam Message Classifier")

# msg = st.text_area("Enter a message")

# if st.button("Predict"):
#     cleaned = simple_clean(msg)
#     vectorized = vectorizer.transform([cleaned])
#     pred = model.predict(vectorized)[0]

#     output = "🚨 SPAM" if pred == 1 else "✔️ NOT SPAM"
#     st.subheader(output)




# app.py
import streamlit as st
import re
import pickle

# -----------------------------
# Simple tokenizer / cleaner
# -----------------------------
def simple_clean(text):
    """
    Cleans text by:
    - Converting to lowercase
    - Removing non-alphabetic characters
    - Tokenizing by splitting on spaces
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    tokens = text.split()                  # simple tokenizer
    return " ".join(tokens)

# -----------------------------
# Load model & vectorizer
# -----------------------------
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's spam or not.")

msg = st.text_area("Message")

if st.button("Predict"):
    if not msg.strip():
        st.warning("Please enter a message!")
    else:
        cleaned = simple_clean(msg)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        output = "🚨 SPAM" if pred == 1 else "✔️ NOT SPAM"
        st.subheader(output)
