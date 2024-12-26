
import pandas as pd
import numpy as np
import sklearn.metrics
from inference import predict

# Load the dataset
balanced_dataset_main_df = pd.read_csv("cleaned_dataset.csv")

# Sample the dataset
balanced_dataset_df = balanced_dataset_main_df.sample(50, random_state=42)

# Apply the prediction function
balanced_dataset_df["response"] = balanced_dataset_df["text"].apply(predict)

# Split the response column into three separate columns
response_split = balanced_dataset_df["response"].str.split(',', expand=True)
balanced_dataset_df["fake_news_predicted"] = response_split[0]
balanced_dataset_df["hate_speech_predicted"] = response_split[1]
balanced_dataset_df["toxicity_predicted"] = response_split[2]

# Ensure no NaN values in the predicted columns
balanced_dataset_df = balanced_dataset_df.dropna(subset=["fake_news_predicted", "hate_speech_predicted", "toxicity_predicted"])

# Convert columns to the same data type
balanced_dataset_df["fake_news_predicted"] = balanced_dataset_df["fake_news_predicted"].astype(int)
balanced_dataset_df["hate_speech_predicted"] = balanced_dataset_df["hate_speech_predicted"].astype(int)
balanced_dataset_df["toxicity_predicted"] = balanced_dataset_df["toxicity_predicted"].astype(int)

# Ensure the true labels are also integers
balanced_dataset_df["fake"] = balanced_dataset_df["fake"].astype(int)
balanced_dataset_df["hatespeech"] = balanced_dataset_df["hatespeech"].astype(int)
balanced_dataset_df["toxicity"] = balanced_dataset_df["toxicity"].astype(int)

# Replace NaN values with a placeholder (e.g., 0 or an empty string) if necessary
balanced_dataset_df = balanced_dataset_df.replace({np.nan: None})

# Convert DataFrame to JSON
json_data = balanced_dataset_df.to_json(orient='records')

# Generate classification reports
print(sklearn.metrics.classification_report(balanced_dataset_df["fake"], balanced_dataset_df["fake_news_predicted"]))
print(sklearn.metrics.classification_report(balanced_dataset_df["hatespeech"], balanced_dataset_df["hate_speech_predicted"]))
print(sklearn.metrics.classification_report(balanced_dataset_df["toxicity"], balanced_dataset_df["toxicity_predicted"]))