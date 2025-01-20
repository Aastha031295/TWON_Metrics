import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def predict(model, dataloader, device):
    """
    Perform inference using the multi-task model.
    """
    model.to(device)
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            fake_news_logits, hate_speech_logits, toxicity_logits = model(input_ids, attention_mask)

            # Apply sigmoid and threshold for binary classification
            fake_news_preds = torch.sigmoid(fake_news_logits).squeeze(1).round().cpu().numpy()
            hate_speech_preds = torch.sigmoid(hate_speech_logits).squeeze(1).round().cpu().numpy()
            toxicity_preds = torch.sigmoid(toxicity_logits).squeeze(1).round().cpu().numpy()

            # Combine predictions
            for fn, hs, tox in zip(fake_news_preds, hate_speech_preds, toxicity_preds):
                predictions.append((fn, hs, tox))

    return predictions
