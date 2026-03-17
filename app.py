import os
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ml_pipeline import (
    load_data,
    preprocess_text_series,
    preprocess_and_split,
    train_model,
    evaluate_model,
)

# ------------------ Page Config ------------------
st.set_page_config(page_title="Xwitter Sentiment Analysis", layout="wide")

# ------------------ Session State Defaults ------------------
if "df" not in st.session_state:
    st.session_state["df"] = None

if "data_source" not in st.session_state:
    st.session_state["data_source"] = None

if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None

if "model" not in st.session_state:
    st.session_state["model"] = None

if "trained_model_name" not in st.session_state:
    st.session_state["trained_model_name"] = None


# ------------------ Kaggle Setup ------------------
def setup_kaggle_credentials():
    if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)

        kaggle_json_path = kaggle_dir / "kaggle.json"
        kaggle_json_content = (
            "{"
            f"\"username\":\"{st.secrets['KAGGLE_USERNAME']}\","
            f"\"key\":\"{st.secrets['KAGGLE_KEY']}\""
            "}"
        )

        kaggle_json_path.write_text(kaggle_json_content, encoding="utf-8")

        try:
            os.chmod(kaggle_json_path, 0o600)
        except Exception:
            pass
    else:
        raise RuntimeError(
            "Kaggle credentials missing. Add KAGGLE_USERNAME and KAGGLE_KEY in Streamlit secrets."
        )


@st.cache_data(show_spinner=False)
def download_kaggle_dataset():
    setup_kaggle_credentials()

    import kaggle

    dataset_dir = Path("data")
    dataset_dir.mkdir(exist_ok=True)

    target_file = dataset_dir / "training.1600000.processed.noemoticon.csv"

    if not target_file.exists():
        kaggle.api.dataset_download_files(
            "kazanova/sentiment140",
            path=str(dataset_dir),
            unzip=True
        )

    if not target_file.exists():
        raise FileNotFoundError("Dataset download hua, lekin CSV file nahi mili.")

    return str(target_file)


# ------------------ Sidebar ------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("Analyze sentiments in tweets using ML models.")

st.sidebar.title("📊 Dataset Info")
st.sidebar.write("Sentiment140 Dataset")
st.sidebar.write("1.6M Tweets (sampled for demo)")


# ------------------ Main Title ------------------
st.title("Xwitter Sentiment Analysis")
st.write("A simple ML-powered app to analyze public mood in tweets.")


# ------------------ Dataset Upload ------------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader(
    "Drag and drop or browse files (CSV)",
    type=["csv"],
    accept_multiple_files=False,
    help="Upload Sentiment140 CSV file",
    key="file_uploader"
)

use_sample = st.button("Use Sample Data", key="sample_button")

# Decide source
selected_source = None
data_path = None

if use_sample:
    selected_source = "sample_data"
    try:
        with st.spinner("Downloading dataset from Kaggle..."):
            data_path = download_kaggle_dataset()
    except Exception as e:
        st.error(f"Sample dataset load failed: {e}")

elif uploaded_file is not None:
    selected_source = f"uploaded::{uploaded_file.name}"
    data_path = uploaded_file

# Load dataset only when source changes or df is missing
if selected_source is not None and data_path is not None:
    try:
        if st.session_state["df"] is None or st.session_state["data_source"] != selected_source:
            with st.spinner("Loading dataset..."):
                st.session_state["df"] = load_data(data_path, sample_size=20000)
                st.session_state["data_source"] = selected_source

                # reset trained objects when dataset changes
                st.session_state["vectorizer"] = None
                st.session_state["model"] = None
                st.session_state["trained_model_name"] = None
    except Exception as e:
        st.error(f"Dataset load failed: {e}")

# ------------------ Dataset Dependent UI ------------------
if st.session_state["df"] is not None:
    df = st.session_state["df"].copy()
    df["clean_text"] = preprocess_text_series(df["text"])

    st.subheader("🔎 Data Preview")
    preview = pd.DataFrame({
        "Original Tweet": df["text"].head(5).values,
        "Cleaned Tweet": df["clean_text"].head(5).values
    })
    st.table(preview)

    # Sentiment counts
    pos_count = (df["polarity"] == 1).sum()
    neg_count = (df["polarity"] == 0).sum()

    st.subheader("📈 Tweet Sentiment Counts")
    st.write(f"Positive Tweets: {pos_count}")
    st.write(f"Negative Tweets: {neg_count}")

    # ------------------ Model Selection ------------------
    st.subheader("🤖 Choose Model")
    model_choice = st.radio(
        "Select a model:",
        ["BernoulliNB", "SVM", "Logistic Regression"],
        key="model_radio"
    )

    train_clicked = st.button(
        "Train Model",
        key=f"train_button_{model_choice}"
    )

    if train_clicked:
        status_box = st.empty()
        result_box = st.empty()

        status_box.info("Training started...")

        X_train, X_test, y_train, y_test, vectorizer = preprocess_and_split(df)

        status_box.info(f"Preprocessing complete. Now training {model_choice}...")
        model = train_model(model_choice, X_train, y_train)

        status_box.info("Training complete. Running evaluation...")
        acc, cm, preds = evaluate_model(model, X_test, y_test)

        status_box.success("Evaluation done.")

        # Save trained objects
        st.session_state["vectorizer"] = vectorizer
        st.session_state["model"] = model
        st.session_state["trained_model_name"] = model_choice

        with result_box.container():
            st.subheader("✅ Model Accuracy")
            st.success(f"{acc * 100:.2f}%")

            st.subheader("📊 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_choice} Confusion Matrix")
            st.pyplot(fig)

    if st.session_state["trained_model_name"] is not None:
        st.info(f"Currently trained model: {st.session_state['trained_model_name']}")

# ------------------ Test a Tweet ------------------
st.subheader("📝 Test a Tweet")
user_tweet = st.text_input("Enter a tweet...", key="tweet_input")

if st.button("Predict Sentiment", key="predict_button"):
    if st.session_state["vectorizer"] is not None and st.session_state["model"] is not None:
        cleaned_tweet = user_tweet.lower().strip()
        tweet_vec = st.session_state["vectorizer"].transform([cleaned_tweet])
        prediction = st.session_state["model"].predict(tweet_vec)[0]

        sentiment = "Positive 😀" if prediction == 1 else "Negative 😡"

        if prediction == 1:
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
    else:
        st.warning("⚠️ Please train a model first before prediction.")