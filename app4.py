import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data():
    true = pd.read_csv('True (1).csv')
    fake = pd.read_csv('Fake.csv')
    fake["label"] = "Fake"
    true["label"] = "True"
    news = pd.concat([fake, true], ignore_index=True)
    news = news.sample(frac=1).reset_index(drop=True)
    news = news.dropna(subset=['text'])
    return news

def preprocess_text(news):
    import re
    import string

    def wordopt(text):
        text = text.lower()
        text = re.sub('\\[.*?\\]', '', text)
        text = re.sub('https?://\\S+|www\\.\\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub(f'[{string.punctuation}]', '', text)
        text = re.sub('\\n', '', text)
        text = re.sub('\\\\w*\\d\\\\w*', '', text)
        return text

    news['text'] = news['text'].apply(wordopt)
    return news

# Train models
def train_models(x_train, y_train):
    vectorisation = TfidfVectorizer()
    xv_train = vectorisation.fit_transform(x_train)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(xv_train, y_train)
        trained_models[name] = model

    return trained_models, vectorisation

def evaluate_model(model, vectorisation, x_test, y_test):
    xv_test = vectorisation.transform(x_test)
    y_pred = model.predict(xv_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, class_report

# Streamlit app
def main():
    st.title("Real or Fake News Detection")

    # Load data
    news = load_data()
    news = preprocess_text(news)

    x = news['text']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(news['label'])  # Convert 'Fake'/'True' to 0/1

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train models
    st.write("Training models... Please wait!")
    trained_models, vectorisation = train_models(x_train, y_train)
    st.success("Training complete!")

    # Sidebar for model selection
    st.sidebar.title("Select a Model")
    model_name = st.sidebar.selectbox("Choose a model", list(trained_models.keys()))
    selected_model = trained_models[model_name]

    # Evaluate selected model
    accuracy, class_report = evaluate_model(selected_model, vectorisation, x_test, y_test)
    st.subheader(f"Performance of {model_name}")
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(class_report)

    # Test custom input
    st.subheader("Test with Custom Input")
    user_input = st.text_area("Enter news text to classify")
    if st.button("Classify"):
        xv_input = vectorisation.transform([user_input])
        prediction = selected_model.predict(xv_input)
        st.write(f"Prediction: {prediction[0]}")

    # Plot model comparison
    st.subheader("Model Performance Comparison")
    accuracies = [evaluate_model(model, vectorisation, x_test, y_test)[0] for model in trained_models.values()]

    fig, ax = plt.subplots()
    sns.barplot(x=list(trained_models.keys()), y=accuracies, palette='viridis', ax=ax)
    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
