"""
This script evaluates the model by comparing the predicted labels with the true labels.
"""

import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import (
    DATA_PATH,
    OUTPUT_FOLDER,
    UNI_MODEL,
    SAMPLE_COUNT,
)
from inference import predict

model_in_use = UNI_MODEL
print(f"Using model: {model_in_use}")

model_in_use_fmt = model_in_use.replace(":", "_").replace("/", "_")

# Load the dataset
balanced_dataset_main_df = pd.read_csv(DATA_PATH)

if SAMPLE_COUNT > 0:
    print(f"Sampling {SAMPLE_COUNT} dataset...")
    balanced_dataset_main_df = balanced_dataset_main_df.sample(SAMPLE_COUNT)

# Apply the prediction function
tqdm.pandas(desc="Applying Classification")
balanced_dataset_main_df["response"] = balanced_dataset_main_df["text"].progress_apply(predict)

# Split the response column into three separate columns
response_split = balanced_dataset_main_df["response"].str.split(",", expand=True)
balanced_dataset_main_df["fake_news_predicted"] = response_split[0]
balanced_dataset_main_df["hate_speech_predicted"] = response_split[1]
balanced_dataset_main_df["toxicity_predicted"] = response_split[2]

# Ensure all label columns are converted to integers
for column in ["fake_news", "hate_speech", "toxicity"]:
    balanced_dataset_main_df[column] = pd.to_numeric(
        balanced_dataset_main_df[column], errors="coerce"
    )

# Replace invalid entries (e.g., NaN) with -1 for the labeled columns
balanced_dataset_main_df.fillna({col: -1 for col in ["fake_news", "hate_speech", "toxicity"]}, inplace=True)

# Convert predictions to integers
############
# Replace invalid entries with 0 for the predicted columns
############
balanced_dataset_main_df["fake_news_predicted"] = pd.to_numeric(
    balanced_dataset_main_df["fake_news_predicted"], errors="coerce"
).fillna(0).astype(int)
balanced_dataset_main_df["hate_speech_predicted"] = pd.to_numeric(
    balanced_dataset_main_df["hate_speech_predicted"], errors="coerce"
).fillna(0).astype(int)
balanced_dataset_main_df["toxicity_predicted"] = pd.to_numeric(
    balanced_dataset_main_df["toxicity_predicted"], errors="coerce"
).fillna(0).astype(int)

# Save predictions to CSV
balanced_dataset_main_df.to_csv(
    f"{OUTPUT_FOLDER}/output_{model_in_use_fmt}.csv", index=False
)

print("Evaluating... ⏳")

# Generate classification reports
fake_report = classification_report(
    balanced_dataset_main_df["fake_news"],
    balanced_dataset_main_df["fake_news_predicted"],
    zero_division=0,
    output_dict=True,
)
fake_report_df = pd.DataFrame(fake_report).transpose()
fake_report_df["category"] = "fake_news"

hatespeech_report = classification_report(
    balanced_dataset_main_df["hate_speech"],
    balanced_dataset_main_df["hate_speech_predicted"],
    zero_division=0,
    output_dict=True,
)
hatespeech_report_df = pd.DataFrame(hatespeech_report).transpose()
hatespeech_report_df["category"] = "hate_speech"

toxicity_report = classification_report(
    balanced_dataset_main_df["toxicity"],
    balanced_dataset_main_df["toxicity_predicted"],
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
