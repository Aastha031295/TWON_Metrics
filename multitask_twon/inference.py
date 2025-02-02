import torch

from tqdm import tqdm
from prompt import CLASSIFICATION_PROMPT  # Ensure the classification prompt is used

def predict(model, dataloader, device, tokenizer, df=None):
    """
    Perform predictions on the dataset.
    """
    model.to(device)
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            # Ensure texts are loaded from dataset
            input_texts = batch["text"]
            input_texts_with_prompt = [f"{CLASSIFICATION_PROMPT} {text}" for text in input_texts]

            # Tokenize with prompt
            encoding = tokenizer(input_texts_with_prompt, padding=True, truncation=True, return_tensors='pt', max_length=128)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Get predictions from model
            fake_news_logits, hate_speech_logits, toxicity_logits = model(input_ids, attention_mask)

            # Apply sigmoid and dynamic thresholding
            threshold = 0.6  # Adjust for better performance
            fake_news_preds = (torch.sigmoid(fake_news_logits) > threshold).squeeze(1).cpu().numpy()
            hate_speech_preds = (torch.sigmoid(hate_speech_logits) > threshold).squeeze(1).cpu().numpy()
            toxicity_preds = (torch.sigmoid(toxicity_logits) > threshold).squeeze(1).cpu().numpy()

            for fn, hs, tox in zip(fake_news_preds, hate_speech_preds, toxicity_preds):
                predictions.append((fn, hs, tox))

    return predictions
