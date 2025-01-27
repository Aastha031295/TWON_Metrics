import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
 
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# Load the dataset
df = pd.read_csv('/home/s2shsinh/TWON_Metrics/unified/merged_dataset_7jan.csv')
 
 #SPlit the dataset
train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)



# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
 
# Custom Dataset Class
class MultiTaskTextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels_fake_news = df['is_fake'].tolist()
        self.labels_hate_speech = df['is_hate_speech'].tolist()
        self.labels_toxicity = df['is_toxic'].tolist()
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return (encoding['input_ids'].squeeze(0),
                encoding['attention_mask'].squeeze(0),
                torch.tensor(self.labels_fake_news[idx], dtype=torch.float32),
                torch.tensor(self.labels_hate_speech[idx], dtype=torch.float32),
                torch.tensor(self.labels_toxicity[idx], dtype=torch.float32))
 
# Create Dataset and DataLoader
# dataset = MultiTaskTextDataset(df)
#Create datasets
train_dataset = MultiTaskTextDataset(train_df)
test_dataset = MultiTaskTextDataset(test_df)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
 
# Define the Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-german-cased')
        self.dropout = nn.Dropout(0.3)
        self.fc_fake_news = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_hate_speech = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_toxicity = nn.Linear(self.bert.config.hidden_size, 1)
 
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Get the pooled output
        pooled_output = self.dropout(pooled_output)
        fake_news_output = self.fc_fake_news(pooled_output)
        hate_speech_output = self.fc_hate_speech(pooled_output)
        toxicity_output = self.fc_toxicity(pooled_output)
        return fake_news_output, hate_speech_output, toxicity_output
 
# Initialize model, optimizer, and loss function
model = MultiTaskModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss(reduction='none')
 
# Training Loop
num_epochs = 3
 
for epoch in range(num_epochs):
    model.train()
    for data in train_dataloader:
        input_ids, attention_mask, labels_fake_news, labels_hate_speech, labels_toxicity = data
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        labels_fake_news, labels_hate_speech, labels_toxicity = labels_fake_news.to(device), labels_hate_speech.to(device), labels_toxicity.to(device)
       
        optimizer.zero_grad()
        outputs_fake_news, outputs_hate_speech, outputs_toxicity = model(input_ids, attention_mask)
       
        loss_fake_news = criterion(outputs_fake_news.squeeze(), labels_fake_news)
        loss_hate_speech = criterion(outputs_hate_speech.squeeze(), labels_hate_speech)
        loss_toxicity = criterion(outputs_toxicity.squeeze(), labels_toxicity)
 
        # Mask the losses where labels are -1
        mask_fake_news = labels_fake_news != -1
        mask_hate_speech = labels_hate_speech != -1
        mask_toxicity = labels_toxicity != -1
       
        loss_fake_news = loss_fake_news[mask_fake_news].mean() if mask_fake_news.any() else torch.tensor(0.0, requires_grad=True).to(device)
        loss_hate_speech = loss_hate_speech[mask_hate_speech].mean() if mask_hate_speech.any() else torch.tensor(0.0, requires_grad=True).to(device)
        loss_toxicity = loss_toxicity[mask_toxicity].mean() if mask_toxicity.any() else torch.tensor(0.0, requires_grad=True).to(device)
 
        total_loss = loss_fake_news + loss_hate_speech + loss_toxicity
        total_loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")
 
# Evaluation
model.eval()
all_labels_fake_news = []
all_preds_fake_news = []
all_labels_hate_speech = []
all_preds_hate_speech = []
all_labels_toxicity = []
all_preds_toxicity = []
 
with torch.no_grad():
    for data in test_dataloader:
        input_ids, attention_mask, labels_fake_news, labels_hate_speech, labels_toxicity = data
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        labels_fake_news, labels_hate_speech, labels_toxicity = labels_fake_news.to(device), labels_hate_speech.to(device), labels_toxicity.to(device)
       
        outputs_fake_news, outputs_hate_speech, outputs_toxicity = model(input_ids, attention_mask)
       
        preds_fake_news = torch.sigmoid(outputs_fake_news).squeeze().round()
        preds_hate_speech = torch.sigmoid(outputs_hate_speech).squeeze().round()
        preds_toxicity = torch.sigmoid(outputs_toxicity).squeeze().round()
       
        mask_fake_news = labels_fake_news != -1
        mask_hate_speech = labels_hate_speech != -1
        mask_toxicity = labels_toxicity != -1
       
        all_labels_fake_news.extend(labels_fake_news[mask_fake_news].cpu().numpy())
        all_preds_fake_news.extend(preds_fake_news[mask_fake_news].cpu().numpy())
       
        all_labels_hate_speech.extend(labels_hate_speech[mask_hate_speech].cpu().numpy())
        all_preds_hate_speech.extend(preds_hate_speech[mask_hate_speech].cpu().numpy())
       
        all_labels_toxicity.extend(labels_toxicity[mask_toxicity].cpu().numpy())
        all_preds_toxicity.extend(preds_toxicity[mask_toxicity].cpu().numpy())
 
accuracy_fake_news = accuracy_score(all_labels_fake_news, all_preds_fake_news)
f1_fake_news = f1_score(all_labels_fake_news, all_preds_fake_news)
 
accuracy_hate_speech = accuracy_score(all_labels_hate_speech, all_preds_hate_speech)
f1_hate_speech = f1_score(all_labels_hate_speech, all_preds_hate_speech)
 
accuracy_toxicity = accuracy_score(all_labels_toxicity, all_preds_toxicity)
f1_toxicity = f1_score(all_labels_toxicity, all_preds_toxicity)
 
print(f"Fake News - Accuracy: {accuracy_fake_news}, F1 Score: {f1_fake_news}")
print(f"Hate Speech - Accuracy: {accuracy_hate_speech}, F1 Score: {f1_hate_speech}")
print(f"Toxicity - Accuracy: {accuracy_toxicity}, F1 Score: {f1_toxicity}")

# Save the model
torch.save(model.state_dict(), '../../media/data/multiTaskTWONB1/multi_task_model.pt')
import torch
import gc
torch.cuda.empty_cache()
gc.collect()