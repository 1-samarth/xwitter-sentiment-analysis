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

    if sample_size is None:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            usecols=[0, 5]
        )
        print("Full dataset loaded.", flush=True)
    else:
        target_per_class = sample_size // 2
        neg_parts = []
        pos_parts = []
        neg_count = 0
        pos_count = 0

        for chunk in pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            usecols=[0, 5],
            chunksize=50000
        ):
            chunk.columns = ["polarity", "text"]

            # keep only 0 and 4
            chunk = chunk[chunk["polarity"].isin([0, 4])].copy()

            neg_chunk = chunk[chunk["polarity"] == 0]
            pos_chunk = chunk[chunk["polarity"] == 4]

            if neg_count < target_per_class and not neg_chunk.empty:
                need_neg = min(target_per_class - neg_count, len(neg_chunk))
                neg_parts.append(neg_chunk.sample(n=need_neg, random_state=42))
                neg_count += need_neg

            if pos_count < target_per_class and not pos_chunk.empty:
                need_pos = min(target_per_class - pos_count, len(pos_chunk))
                pos_parts.append(pos_chunk.sample(n=need_pos, random_state=42))
                pos_count += need_pos

            if neg_count >= target_per_class and pos_count >= target_per_class:
                break

        if neg_count == 0 or pos_count == 0:
            raise ValueError(
                f"Balanced sample create nahi hua. Negative: {neg_count}, Positive: {pos_count}"
            )

        df = pd.concat(neg_parts + pos_parts, ignore_index=True)
        print(
            f"Collected balanced sample -> Negative: {neg_count}, Positive: {pos_count}",
            flush=True
        )

    if "polarity" not in df.columns or "text" not in df.columns:
        df.columns = ["polarity", "text"]

    # Convert labels: 0 = negative, 4 = positive -> 0, 1
    df["polarity"] = df["polarity"].map({0: 0, 4: 1})

    # Null values remove
    df = df.dropna(subset=["text", "polarity"])

    # Final shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Class distribution:", df["polarity"].value_counts().to_dict(), flush=True)
    print(f"Final usable rows: {len(df)}", flush=True)

    return df


def preprocess_text_series(text_series):
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
        model = LinearSVC(max_iter=3000)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
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