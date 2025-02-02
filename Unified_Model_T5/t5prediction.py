import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=128, legacy=False)

# Load the trained model
class MultiTaskT5(torch.nn.Module):
    def __init__(self):
        super(MultiTaskT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    def forward(self, input_ids, attention_mask):
        return self.t5.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=20)

# Initialize model and load weights
model = MultiTaskT5().to(device)
model.load_state_dict(torch.load('t5_multi_task_model.pt', map_location=device))
model.eval()

# Load the dataset
df = pd.read_csv('merged_dataset_31stJan.csv')

# Custom Dataset Class for Prediction
class MultiTaskTextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].astype(str).tolist()  # Ensure all text values are strings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(
            text, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

# Create Dataset and DataLoader for Prediction
dataset = MultiTaskTextDataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(torch.nn.utils.rnn.pad_sequence([i[j] for i in x], batch_first=True) for j in range(2)))

# Perform Prediction
preds = []
with torch.no_grad():
    for input_ids, attention_mask in dataloader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded_preds)

# Convert text predictions to numerical labels
def extract_labels(output_texts):
    extracted = []
    for text in output_texts:
        extracted_values = {"fake_news": -1, "hate_speech": -1, "toxicity": -1}
        matches = re.findall(r"(fake_news|hate_speech|toxicity):\s*(0|1)", text)
        for match in matches:
            category, value = match
            extracted_values[category] = int(value)
        extracted.append(extracted_values)
    return extracted

# Process Predictions
processed_preds = extract_labels(preds)

# Convert to DataFrame and Save
predictions_df = pd.DataFrame(processed_preds)
df.reset_index(drop=True, inplace=True)  # Ensure indices align
df_predictions = pd.concat([df, predictions_df], axis=1)

# Save to CSV
output_path = "predictions_T5.csv"
df_predictions.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")
