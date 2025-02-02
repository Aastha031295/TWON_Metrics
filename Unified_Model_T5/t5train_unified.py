import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv('merged_dataset_31stJan.csv')

# Split dataset into training and testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=128, legacy=False)

# Custom Dataset Class
class MultiTaskTextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].astype(str).tolist()  # Ensure all text values are strings
        self.labels = [
            f"fake_news: {int(row['fake_news'])} hate_speech: {int(row['hate_speech'])} toxicity: {int(row['toxicity'])}"
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Convert to string to avoid tokenization errors
        label = str(self.labels[idx])  # Ensure labels are also valid strings
        
        encoding = tokenizer(
            text, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )
        label_encoding = tokenizer(
            label, return_tensors='pt', padding='max_length', truncation=True, max_length=32
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            label_encoding['input_ids'].squeeze(0),
        )

# Create Dataset and DataLoader
train_dataset = MultiTaskTextDataset(train_df)
test_dataset = MultiTaskTextDataset(test_df)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(torch.nn.utils.rnn.pad_sequence([i[j] for i in x], batch_first=True) for j in range(3)))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(torch.nn.utils.rnn.pad_sequence([i[j] for i in x], batch_first=True) for j in range(3)))

# Define T5 Model
class MultiTaskT5(torch.nn.Module):
    def __init__(self):
        super(MultiTaskT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    def forward(self, input_ids, attention_mask, labels):
        return self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Initialize model, optimizer, and scheduler
model = MultiTaskT5().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1.5e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 7)

# Training Loop
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_dataloader:
        batch = [b.to(device) for b in [input_ids, attention_mask, labels]]
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")

# Evaluation
model.eval()
preds, labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, label_ids = batch
        outputs = model.t5.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=20)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        preds.extend(decoded_preds)
        labels.extend(decoded_labels)

# Convert text predictions to numerical labels

def extract_labels(output_texts):
    extracted = []
    for text in output_texts:
        extracted_values = {"fake_news": -1, "hate_speech": -1, "toxicity": -1}
        matches = re.findall(r"(fake_news|hate_speech|toxicity):\s*(0|1)", text)
        for match in matches:
            category, value = match
            extracted_values[category] = int(value)
        extracted.append(list(extracted_values.values()))
    return np.array(extracted)

true_labels = extract_labels(labels)
pred_labels = extract_labels(preds)

# Ensure consistent label shape
if true_labels.shape == pred_labels.shape:
    for i, category in enumerate(["Fake News", "Hate Speech", "Toxicity"]):
        accuracy = accuracy_score(true_labels[:, i], pred_labels[:, i])
        precision = precision_score(true_labels[:, i], pred_labels[:, i], average='macro', zero_division=1)
        recall = recall_score(true_labels[:, i], pred_labels[:, i], average='macro', zero_division=1)
        f1 = f1_score(true_labels[:, i], pred_labels[:, i], average='macro')
        print(f"{category} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
else:
    print("Fixed shape mismatch by ensuring all predictions have three numerical values.")

# Save the model
torch.save(model.state_dict(), 't5_multi_task_model.pt')
