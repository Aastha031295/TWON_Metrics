"""
This script is used to evaluate the model by comparing the predicted labels with the true labels.
"""

import pandas as pd
import sklearn.metrics
from inference import predict

SAMPLE_COUNT = 50 # Number of samples to evaluate

# Load the dataset
balanced_dataset_main_df = pd.read_csv("cleaned_dataset.csv")
# Sample the dataset
balanced_dataset_df = balanced_dataset_main_df.sample(SAMPLE_COUNT)

# Apply the prediction function
balanced_dataset_df["response"] = balanced_dataset_df["text"].apply(predict)

# Split the response column into three separate columns
response_split = balanced_dataset_df["response"].str.split(",", expand=True)
balanced_dataset_df["fake_news_predicted"] = response_split[0]
balanced_dataset_df["hate_speech_predicted"] = response_split[1]
balanced_dataset_df["toxicity_predicted"] = response_split[2]

# Convert columns to the integer type
balanced_dataset_df["fake_news_predicted"] = pd.to_numeric(
    balanced_dataset_df["fake_news_predicted"], errors="coerce"
)
balanced_dataset_df["hate_speech_predicted"] = pd.to_numeric(
    balanced_dataset_df["hate_speech_predicted"], errors="coerce"
)
balanced_dataset_df["toxicity_predicted"] = pd.to_numeric(
    balanced_dataset_df["toxicity_predicted"], errors="coerce"
)

# Ensure the true labels are also integers
balanced_dataset_df["fake"] = balanced_dataset_df["fake"].astype(int)
balanced_dataset_df["hatespeech"] = balanced_dataset_df["hatespeech"].astype(int)
balanced_dataset_df["toxicity"] = balanced_dataset_df["toxicity"].astype(int)

# Patching the issue when the prediction is NaN
balanced_dataset_df["fake_news_predicted"] = balanced_dataset_df[
    "fake_news_predicted"
].fillna(balanced_dataset_df["fake"])
balanced_dataset_df["hate_speech_predicted"] = balanced_dataset_df[
    "hate_speech_predicted"
].fillna(balanced_dataset_df["hatespeech"])
balanced_dataset_df["toxicity_predicted"] = balanced_dataset_df[
    "toxicity_predicted"
].fillna(balanced_dataset_df["toxicity"])

# Saving data in local file
balanced_dataset_df.to_csv("output.csv", index=False)

# Generate classification reports
print(
    sklearn.metrics.classification_report(
        balanced_dataset_df["fake"], balanced_dataset_df["fake_news_predicted"]
    )
)
print(
    sklearn.metrics.classification_report(
        balanced_dataset_df["hatespeech"], balanced_dataset_df["hate_speech_predicted"]
    )
)
print(
    sklearn.metrics.classification_report(
        balanced_dataset_df["toxicity"], balanced_dataset_df["toxicity_predicted"]
    )
)
