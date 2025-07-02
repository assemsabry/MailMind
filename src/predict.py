from joblib import load
from preprocess import clean_text

def predict_with_roberta(text):
    clf, vectorizer = load("models/roberta_email_model.pkl")
    cleaned = clean_text(text)
    vector = vectorizer.vectorize([cleaned])
    return clf.predict(vector)[0]
