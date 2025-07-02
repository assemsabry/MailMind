import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["message", "label"])
    df["clean_text"] = df["message"].apply(clean_text)
    return df[["clean_text", "label"]]