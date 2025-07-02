import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from preprocess import preprocess_dataset
from bert_vectorizer import BERTVectorizer

def train_with_roberta():
    df = preprocess_dataset("data/processed/emails.csv")
    X = df["clean_text"]
    y = df["label"]

    print("\n🔍 Vectorizing using RoBERTa...")
    vectorizer = BERTVectorizer("roberta-base")
    X_vec = vectorizer.vectorize(X.tolist())

    print("\n📊 Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    dump((clf, vectorizer), "models/roberta_email_model.pkl")
    print("\n✅ Model saved to models/roberta_email_model.pkl")

if __name__ == "__main__":
    train_with_roberta()