import os
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_model, load_tokenizer
from dataset import SocialMediaDataset, load_dataset
from inference import predict
from sklearn.metrics import classification_report
from config import DATA_PATH, OUTPUT_PATH, DEVICE, MAX_LEN, BATCH_SIZE

def evaluate(predictions, df):
    """
    Evaluate predictions against true labels.
    """
    categories = ["fake_news", "hate_speech", "toxicity"]
    evaluation_reports = []

    for i, category in enumerate(categories):
        valid_indices = df[category] != -1
        y_true = df[category][valid_indices]
        y_pred = [pred[i] for idx, pred in enumerate(predictions) if valid_indices.iloc[idx]]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df["category"] = category
        evaluation_reports.append(report_df)

    combined_report = pd.concat(evaluation_reports, ignore_index=False)
    return combined_report

def main():
    # Ensure correct dataset is loaded
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_model(DEVICE)

    # Prepare dataset and DataLoader
    print("Preparing dataset...")
    dataset = SocialMediaDataset(df, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Run inference
    print("Running inference...")
    predictions = predict(model, dataloader, DEVICE, tokenizer, df)

    # Save predictions in DataFrame
    df["fake_news_predicted"], df["hate_speech_predicted"], df["toxicity_predicted"] = zip(*predictions)

    # Save predictions to CSV
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, "predictions.csv")
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Evaluate predictions
    print("Evaluating predictions...")
    evaluation_report = evaluate(predictions, df)

    # Save evaluation report
    evaluation_file = os.path.join(OUTPUT_PATH, "evaluation_report.csv")
    evaluation_report.to_csv(evaluation_file, index=True)
    print(f"Evaluation report saved to {evaluation_file}")

if __name__ == "__main__":
    main()
