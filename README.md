# 🐦 Xwitter Sentiment Analysis

A machine learning web app to analyze sentiment (positive/negative) of tweets using multiple ML models.

---

## 🔗 Live App
👉 (yaha apna Streamlit app link daalna)

---

## 📌 Features

- 📂 Upload your own CSV dataset
- 📥 Upload Xquik/export CSV files with tweet text and sentiment labels
- 📊 Use Kaggle Sentiment140 dataset (auto download)
- 🧹 Text preprocessing (lowercase, cleaning)
- 🤖 Multiple ML models:
  - Bernoulli Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- 📈 Model evaluation with:
  - Accuracy score
  - Confusion Matrix
- 📝 Real-time tweet sentiment prediction

---

## 📊 Dataset

- **Name:** Sentiment140 Dataset  
- **Source:** Kaggle  
- **Size:** 1.6 million tweets

👉 Dataset automatically downloaded using Kaggle API

You can also upload Xquik exports or other CSV files that include one tweet
text column (`text`, `tweet`, `tweet_text`, `content`, `full_text`, or
`message`) and one label column (`sentiment`, `polarity`, `label`, or
`target`). Labels can use `positive` / `negative`, `pos` / `neg`, or the
Sentiment140-style `4` / `0` values.

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib & Seaborn
- Kaggle API

---

## 🧠 How It Works

1. Dataset is loaded (Kaggle / Upload)
2. Text is preprocessed
3. TF-IDF vectorization is applied
4. Model is trained
5. Model is evaluated
6. User can test custom tweets

