import streamlit as st
st.set_page_config(
    page_title="Interactive Multitask Prediction App",
    layout="wide",
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Define the Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-german-cased')
        self.dropout = nn.Dropout(0.3)
        self.fc_fake_news = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_hate_speech = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_toxicity = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        fake_news_output = self.fc_fake_news(pooled_output)
        hate_speech_output = self.fc_hate_speech(pooled_output)
        toxicity_output = self.fc_toxicity(pooled_output)
        return fake_news_output, hate_speech_output, toxicity_output

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = MultiTaskModel()
    model.load_state_dict(torch.load('multi_task_model.pt', map_location=device))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    return model, tokenizer

model, tokenizer = load_model()

# Function to make predictions
def predict_text(texts):
    predictions = []
    for text in texts:
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs_fake_news, outputs_hate_speech, outputs_toxicity = model(input_ids, attention_mask)

        preds_fake_news = torch.sigmoid(outputs_fake_news).squeeze().item()
        preds_hate_speech = torch.sigmoid(outputs_hate_speech).squeeze().item()
        preds_toxicity = torch.sigmoid(outputs_toxicity).squeeze().item()

        predictions.append({
            "Text": text,
            "Fake News Probability": preds_fake_news,
            "Hate Speech Probability": preds_hate_speech,
            "Toxicity Probability": preds_toxicity
        })

    return pd.DataFrame(predictions)

# App Layout
st.title("Interactive Multitask Prediction App")

# User Input Section
st.header("Input Text or Dataset")
input_mode = st.radio("Select Input Mode:", ("Single Text Input", "Upload Dataset"))

if input_mode == "Single Text Input":
    user_text = st.text_area("Enter your text here:")
    if st.button("Predict Text"):
        if user_text.strip():
            result_df = predict_text([user_text])
            st.write("### Prediction Results")
            st.dataframe(result_df)

            # Download results as Excel
            download_path = st.text_input("Enter file path to save results (e.g., predictions.xlsx):")
            if st.button("Save Results to Excel"):
                if download_path:
                    result_df.to_excel(download_path, index=False)
                    st.success(f"Results saved to {download_path}")
                else:
                    st.error("Please specify a valid file path.")
        else:
            st.error("Please enter text for prediction.")

elif input_mode == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload your dataset (CSV, TXT, etc.):", type=["csv", "txt", "text"])
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            text_data = uploaded_file.read().decode("utf-8")
            texts = text_data.splitlines()
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            texts = df.iloc[:, 0].tolist()  # Assuming the first column contains the text
        else:
            st.error("Unsupported file format.")
            texts = []

        if texts:
            st.write(f"Loaded {len(texts)} entries for prediction.")
            if st.button("Predict Dataset"):
                result_df = predict_text(texts)
                st.write("### Prediction Results")
                st.dataframe(result_df)

                # Download results as Excel
                download_path = st.text_input("Enter file path to save results (e.g., predictions.xlsx):")
                if st.button("Save Results to Excel"):
                    if download_path:
                        result_df.to_excel(download_path, index=False)
                        st.success(f"Results saved to {download_path}")
                    else:
                        st.error("Please specify a valid file path.")

# Analysis Section
st.header("Results Analysis")
if 'result_df' in locals() or 'result_df' in globals():
    st.write("### Visualizations")

    # Probabilities Distribution
    st.write("#### Distribution of Predictions")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(result_df['Fake News Probability'], bins=20, color='blue', alpha=0.7)
    ax[0].set_title("Fake News Probability")
    ax[1].hist(result_df['Hate Speech Probability'], bins=20, color='red', alpha=0.7)
    ax[1].set_title("Hate Speech Probability")
    ax[2].hist(result_df['Toxicity Probability'], bins=20, color='green', alpha=0.7)
    ax[2].set_title("Toxicity Probability")
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("#### Correlation Between Predictions")
    corr = result_df.drop(columns="Text").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.colorbar(cax)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)
