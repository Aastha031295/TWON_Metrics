import pandas as pd
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


# Load your dataset
df = pd.read_csv("/Users/shivangsinha/Downloads/ROCMetrics/predictionsFinal25jan2025-test2.csv")

# # Group by the three columns
# grouped = df.groupby(['isFake', 'isHateSpeech', 'isToxic'])

# # Sample 1000 records from each group
# sampled_df = grouped.apply(lambda x: x.sample(n=1000, replace=False if len(x) >= 1000 else True)).reset_index(drop=True)

# # Check the final dataset
# print(sampled_df['isFake'].value_counts())
# print(sampled_df['isHateSpeech'].value_counts())
# print(sampled_df['isToxic'].value_counts())

# # Save the balanced dataset
# sampled_df.to_csv("balanced_dataset.csv", index=False)


# Extract true and predicted labels

# Filter out rows where true_isFake is -1
valid_fake_news = df[df['isFake'] != -1]
valid_hate_speech = df[df['isHateSpeech'] != -1]
valid_toxicity = df[df['isToxic'] != -1]


true_isFake = valid_fake_news['isFake']
pred_isFake = valid_fake_news['predicted_isFake']

true_isHateSpeech = valid_hate_speech['isHateSpeech']
pred_isHateSpeech = valid_hate_speech['predicted_isHateSpeech']

true_isToxic = valid_toxicity['isToxic']
pred_isToxic = valid_toxicity['predicted_isToxic']

# Function to calculate and display metrics
def calculate_metrics(true, pred, label_name):
    print(f"Metrics for {label_name}:")
    print(f"Accuracy: {accuracy_score(true, pred):.2f}")
    print(f"F1 Score: {f1_score(true, pred, average='binary', pos_label=1):.2f}")
    print(f"Precision: {precision_score(true, pred, average='binary', pos_label=1):.2f}")
    print(f"Recall: {recall_score(true, pred, average='binary', pos_label=1):.2f}")
    print("\n")

# Calculate metrics for each label
calculate_metrics(true_isFake, pred_isFake, "isFake")
calculate_metrics(true_isHateSpeech, pred_isHateSpeech, "isHateSpeech")
calculate_metrics(true_isToxic, pred_isToxic, "isToxic")