import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(path, sample_size=None):
    print("Loading dataset...", flush=True)

    # Sirf required columns read karo
    # sample_size diya ho to utni hi rows load karo
    if sample_size is not None:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            usecols=[0, 5],
            nrows=sample_size
        )
        print(f"Loaded first {sample_size} rows for faster training.", flush=True)
    else:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            usecols=[0, 5]
        )
        print("Full dataset loaded.", flush=True)

    df.columns = ["polarity", "text"]

    # Remove neutral tweets
    df = df[df["polarity"] != 2].copy()

    # Convert labels: 0 = negative, 4 = positive -> 0, 1
    df["polarity"] = df["polarity"].map({0: 0, 4: 1})

    # Null values remove
    df = df.dropna(subset=["text", "polarity"])

    print(f"Final usable rows: {len(df)}", flush=True)

    return df


def preprocess_text_series(text_series):
    # Basic cleaning
    return text_series.astype(str).str.lower().str.strip()


def preprocess_and_split(df):
    print("Preprocessing started...", flush=True)

    df = df.copy()
    df["clean_text"] = preprocess_text_series(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["polarity"],
        test_size=0.2,
        random_state=42,
        stratify=df["polarity"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Preprocessing complete.", flush=True)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_model(model_name, X_train, y_train):
    print(f"Training started with {model_name}...", flush=True)
    start = time.time()

    if model_name == "BernoulliNB":
        model = BernoulliNB()
    elif model_name == "SVM":
        model = LinearSVC(max_iter=2000)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.fit(X_train, y_train)

    end = time.time()
    print(f"Training complete! Time taken: {end - start:.2f} seconds", flush=True)

    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluation started...", flush=True)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Evaluation done. Accuracy: {acc * 100:.2f}%", flush=True)

    return acc, cm, preds