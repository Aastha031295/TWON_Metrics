# import pandas
# import sklearn.metrics

# import cltrier_lib

# DATA_FILE: str = "/home/s2shsinh/data/processed/DefaktS_Twitter.binary.csv"
# N_SAMPLES: int = 500

# dataset: pandas.DataFrame = (
#     pandas.read_csv(DATA_FILE, index_col=[0])
#     .replace(dict(binary_label={0.0: "neutral_post", 1.0: "possible_fake_news"}))
#     .sample(n=N_SAMPLES)
# )
# print(dataset.head())

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np

# Load Dataset
import pandas as pd

# Example CSV File
data: str = "/home/s2shsinh/TWON_Metrics/dataset/balanced_dataset.csv"

# data = {
#     "fake": [0, 1, 1],
#     "hatespeech": [0, 0, 0],
#     "toxicity": [0, 0, 0],
#     "cleaned_text": [
#         "studio satlive sprechen heute stefan schulte vorstandsvorsitzenden fraportag lage frankfurter flughafen bundespräsident steinmeier übergibt mainz welterbeurkunde unesco frühere zentren jüdischer gelehrsamkeit rlp",
#         "frauen amp männer iran wunsch freiheit macht gerne unterschreibt irgcterrorists klimaaktivisten",
#         "politikerinnen samt linksgrünemedien örr tautologie markussoeder csu g traditionellen tracht bayern ausgelacht diskrimiert widerlich doppelmoral",
#     ]
# }
df = pd.read_csv(data)

# Preprocessing Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        # labels = torch.tensor([row["fake"], row["hatespeech"], row["toxicity"]], dtype=torch.float)
        labels = torch.tensor([row["binary_label"]], dtype=torch.float)
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

# # Initialize Tokenizer and Dataset

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = MultiTaskDataset(df, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define Multi-Task Model
class MultiTaskBERT(nn.Module):
    def __init__(self, pretrained_model_name):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=1)
        self.fake_news_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.hatespeech_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.toxicity_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        fake_news_pred = torch.sigmoid(self.fake_news_head(pooled_output))
        hatespeech_pred = torch.sigmoid(self.hatespeech_head(pooled_output))
        toxicity_pred = torch.sigmoid(self.toxicity_head(pooled_output))
        return fake_news_pred, hatespeech_pred, toxicity_pred

# # Initialize Model
model = MultiTaskBERT(pretrained_model_name="bert-base-uncased")
# optimizer = AdamW(model.parameters(), lr=5e-5)
# criterion = nn.BCELoss()

# Training Loop
# model.train()
# epochs = 3
# for epoch in range(epochs):
#     total_loss = 0
#     for batch in dataloader:
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]

#         # Forward pass
#         fake_news_pred, hatespeech_pred, toxicity_pred = model(input_ids, attention_mask)

#         # Calculate loss for each task
#         fake_news_loss = criterion(fake_news_pred.squeeze(), labels[:, 0])
#         hatespeech_loss = criterion(hatespeech_pred.squeeze(), labels[:, 0])
#         toxicity_loss = criterion(toxicity_pred.squeeze(), labels[:, 0])

#         # Combine losses
#         loss = (fake_news_loss + hatespeech_loss + toxicity_loss) / 3

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# # Save the Model
# torch.save(model.state_dict(), "multi_task_bert.pth")
# print("Model saved as multi_task_bert.pth")

# Load the Model for Prediction
model.load_state_dict(torch.load("multi_task_bert.pth"))
model.eval()

# Example Prediction
def predict(text):
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        fake_news_pred, hatespeech_pred, toxicity_pred = model(
            inputs["input_ids"], inputs["attention_mask"]
        )
    return {
        "fake_news": fake_news_pred.item(),
        "hate_speech": hatespeech_pred.item(),
        "toxicity": toxicity_pred.item(),
    }

# Test Prediction
sample_text = "Die Frauen &amp; Männer im #Iran haben einen Wunsch: Freiheit! Macht gerne mit und unterschreibt. #IRGCterrorists Klimaaktivisten "
predictions = predict(sample_text)
print("Predictions:", predictions)
