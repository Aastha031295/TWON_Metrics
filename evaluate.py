"""
This script is used to evaluate the model by comparing the predicted labels with the true labels.
"""

import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import (
    DATA_PATH,
    HG_MODEL,
    OUTPUT_FOLDER,
    SAMPLE_COUNT,
    UNI_MODEL,
    USE_UNI_LLM_API,
)
from inference import predict

model_in_use = UNI_MODEL if USE_UNI_LLM_API else HG_MODEL
print(f"Using model: {model_in_use}")
print(f"Sample count: {SAMPLE_COUNT}")

model_in_use_fmt = model_in_use.replace(":", "_").replace("/", "_")

# Load the dataset
balanced_dataset_main_df = pd.read_csv(DATA_PATH)
# Sample the dataset
balanced_dataset_df = balanced_dataset_main_df.sample(SAMPLE_COUNT)

# Apply the prediction function
tqdm.pandas(desc="Applying Classification")
balanced_dataset_df["response"] = balanced_dataset_df["text"].progress_apply(predict)

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
balanced_dataset_df.loc[:, "fake_news"] = balanced_dataset_df["fake_news"].astype(int)
balanced_dataset_df.loc[:, "hate_speech"] = balanced_dataset_df["hate_speech"].astype(int)
balanced_dataset_df.loc[:, "toxicity"] = balanced_dataset_df["toxicity"].astype(int)

# Patching the issue when the prediction is NaN
balanced_dataset_df = balanced_dataset_df.fillna(
    {"fake_news_predicted": -1, "hate_speech_predicted": -1, "toxicity_predicted": -1}
)

# Saving data in local file
balanced_dataset_df.to_csv(
    f"{OUTPUT_FOLDER}/output_{model_in_use_fmt}.csv", index=False
)

print("Evaluating... ⏳")
# Generate classification reports
fake_report = classification_report(
    balanced_dataset_df["fake_news"],
    balanced_dataset_df["fake_news_predicted"],
    zero_division=0,
    output_dict=True,
)
fake_report_df = pd.DataFrame(fake_report).transpose()
fake_report_df["category"] = "fake_news"

hatespeech_report = classification_report(
    balanced_dataset_df["hate_speech"],
    balanced_dataset_df["hate_speech_predicted"],
    zero_division=0,
    output_dict=True,
)
hatespeech_report_df = pd.DataFrame(hatespeech_report).transpose()
hatespeech_report_df["category"] = "hate_speech"

toxicity_report = classification_report(
    balanced_dataset_df["toxicity"],
    balanced_dataset_df["toxicity_predicted"],
    zero_division=0,
    output_dict=True,
)
toxicity_report_df = pd.DataFrame(toxicity_report).transpose()
toxicity_report_df["category"] = "toxicity"

# Concatenate all reports into a single DataFrame
combined_report_df = pd.concat(
    [fake_report_df, hatespeech_report_df, toxicity_report_df]
)

# Save the combined report to a CSV file
combined_report_df.to_csv(f"{OUTPUT_FOLDER}/eval_{model_in_use_fmt}.csv", index=True)

print(f"Evaluation completed. Check the output files ({model_in_use_fmt}).")
