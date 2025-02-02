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

# Load Dataset
balanced_dataset_main_df = pd.read_csv(DATA_PATH)

# Sample Data (if required)
if SAMPLE_COUNT > 0:
    print(f"Sampling {SAMPLE_COUNT} dataset...")
    balanced_dataset_main_df = balanced_dataset_main_df.sample(SAMPLE_COUNT)

# Apply Prediction Function
tqdm.pandas(desc="Applying Classification")
balanced_dataset_main_df["response"] = balanced_dataset_main_df["text"].progress_apply(predict)

# Extract Predictions
response_split = balanced_dataset_main_df["response"].str.split(",", expand=True)
balanced_dataset_main_df["fake_news_predicted"] = response_split[0]
balanced_dataset_main_df["hate_speech_predicted"] = response_split[1]
balanced_dataset_main_df["toxicity_predicted"] = response_split[2]

# Convert labels to numeric
for column in ["fake_news", "hate_speech", "toxicity"]:
    balanced_dataset_main_df[column] = pd.to_numeric(
        balanced_dataset_main_df[column], errors="coerce"
    )

# Convert Predictions to Numeric & Fill NaN with 0
for col in ["fake_news_predicted", "hate_speech_predicted", "toxicity_predicted"]:
    balanced_dataset_main_df[col] = pd.to_numeric(
        balanced_dataset_main_df[col], errors="coerce"
    ).fillna(0).astype(int)

# Save Predictions
balanced_dataset_main_df.to_csv(
    f"{OUTPUT_FOLDER}/output_{model_in_use_fmt}.csv", index=False
)

print("Evaluating... ⏳")

# Function for `-1` Handling per Feature
def evaluate_feature(true_col, pred_col, category):
    """
    Evaluates a single feature while ignoring `-1` values.
    """
    valid_indices = balanced_dataset_main_df[true_col] != -1  # Ignore only for this feature
    y_true = balanced_dataset_main_df.loc[valid_indices, true_col]  # Valid labels (0,1)
    y_pred = balanced_dataset_main_df.loc[valid_indices, pred_col]  # Corresponding predictions

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["category"] = category
    return report_df

# Compute Reports with `-1` Handling
fake_news_report = evaluate_feature("fake_news", "fake_news_predicted", "fake_news")
hate_speech_report = evaluate_feature("hate_speech", "hate_speech_predicted", "hate_speech")
toxicity_report = evaluate_feature("toxicity", "toxicity_predicted", "toxicity")

# Merge Reports
combined_report_df = pd.concat([fake_news_report, hate_speech_report, toxicity_report])

# Save Evaluation Report
combined_report_df.to_csv(f"{OUTPUT_FOLDER}/eval_{model_in_use_fmt}.csv", index=True)

print(f" Evaluation Completed. Check output files: {model_in_use_fmt}.")
