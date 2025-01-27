import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load your CSV file
# Replace 'your_file.csv' with the actual file path
df = pd.read_csv('/home/s2shsinh/TWON_Metrics/unified/predictionsFinal.csv')

# Define a function to calculate metrics for a specific column
def calculate_metrics(df, ground_truth_col, predicted_col):
    # Filter rows where the ground truth is not -1
    filtered_df = df[df[ground_truth_col] != -1]
    
    # Extract ground truth and predictions
    y_true = filtered_df[ground_truth_col]
    y_pred = filtered_df[predicted_col]
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average='binary'),
        "Precision": precision_score(y_true, y_pred, average='binary'),
        "Recall": recall_score(y_true, y_pred, average='binary')
    }
    return metrics

# Calculate metrics for each column pair
metrics_isFake = calculate_metrics(df, 'isFake', 'predicted_isFake')
metrics_isHateSpeech = calculate_metrics(df, 'isHateSpeech', 'predicted_isHateSpeech')
metrics_isToxic = calculate_metrics(df, 'isToxic', 'predicted_isToxic')

# Print the results
print("Metrics for Is Fake:")
print(metrics_isFake)

print("\nMetrics for Is Hate Speech:")
print(metrics_isHateSpeech)

print("\nMetrics for Is Toxic:")
print(metrics_isToxic)
