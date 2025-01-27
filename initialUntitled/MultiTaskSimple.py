import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score

# Load datasets
fake_news_df = pd.read_csv('fake_news.csv')
hate_speech_df = pd.read_csv('hate_speech.csv')
toxicity_df = pd.read_csv('toxicity.csv')

# Merge datasets on tweet_id or some common identifier
merged_df = fake_news_df.merge(hate_speech_df, on=['tweet_id', 'tweet_text'], how='outer')
merged_df = merged_df.merge(toxicity_df, on=['tweet_id', 'tweet_text'], how='outer')

# Fill missing Labels with @ (or any placeholder you prefer)
merged_df[ 'fake_news'] = merged_df['fake_news' ].fillna(0)
merged_df['hate _speech'] = merged_df['hate_speech'].fillna(0)
merged_df[ 'toxicity'] = merged_df['toxicity']. fillna(0)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')

# Custom Dataset class
class MultiTaskDataset (Dataset):
    def _init_(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tweet = str(self.data.tweet_text[index])
        fake_news = self.data.fake_news[index]
        hate_speech = self.data.hate_speech[index]
        toxicity = self.data.toxicity[index]
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=False, 
            truncation=True
        )
        

        input_ids=inputs[' input_ids']
        attention_mask = inputs['attention _mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': {
                'fake_news': torch.tensor(fake_news, dtype=torch.float),
                'hate_speech': torch. tensor (hate_speech, dtype=torch.float),
                'toxicity': torch. tensor(toxicity, dtype=torch.float)
            }
        }
        
# Define the multi-task model
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super (MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.fake_news_classifier = nn.Linear (base_model.config.hidden_size, 1) 
        self.hate_speech_classifier = nn.Linear (base_model.config.hidden_size, 1) 
        self.toxicity_classifier = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model (input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] # CLS token representation

        fake_news_logits = self. fake_news_classifier(pooled_output)
        hate_speech_logits = self.hate_speech_classifier(pooled_output)
        toxicity_logits = self.toxicity_classifier(pooled_output)
        
        return fake_news_logits, hate_speech_logits, toxicity_logits
    
# Initialize BERT model
base_model = BertModel.from_pretrained('bert-base-uncased')
multi_task_model = MultiTaskModel(base_model)

# Define Loss functions and optimizer
criterion = nn.BCEWithLogitsLoss()

# Define hyperparameter grid
param_grid ={
    'learning_rate' : [1e-5, 2e-5, 3e-5],
    'batch_size': [8, 16, 32],
    'weight_decay': [0.0, 0.01, 0.1],
    'num_epochs': [3, 4, 5]
}


# Create parameter combinations
param_combinations = list(ParameterGrid (param_grid))
# Function to evaluate model
def evaluate_model (model, dataloader):
    model.eval()
    all_preds = {'fake_news': [], 'hate_speech': [], 'toxicity': []} 
    all_labels = {'fake_news': [], 'hate_speech': [], 'toxicity': []}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[' input_ids']
            attention_mask = batch[ 'attention _mask']
            labels = batch['labels']
            fake_news_logits, hate_speech_logits, toxicity_logits = model(input_ids, attention_mask)
            fake_news_preds = (torch.sigmoid(fake_news_logits) > 0.5).int()
            hate_speech_preds = (torch.sigmoid(hate_speech_logits) > 0.5).int()
            toxicity_preds = (torch.sigmoid(toxicity_logits) > 0.5).int()
            all_preds [' fake_news' ].extend(fake_news_preds.cpu().numpy())
            all_preds ['hate_speech'].extend(hate_speech_preds.cpu().numpy())
            all_preds ['toxicity'].extend (toxicity_preds.cpu().numpy ())
            all_labels['fake_news'].extend (labels[' fake_news '].cpu().numpy())

            all_labels[ 'hate_speech'].extend(labels['hate_speech'].cpu() .numpy ())
            all_labels[ 'toxicity'].extend(labels[ 'toxicity'].cpu().numpy())
    # Calculate accuracy for each task
    accuracies = {
    'fake_news': accuracy_score(all_labels['fake_news'], all_preds['fake_news']),
    'hate_speech': accuracy_score(all_labels['hate_speech'], all_preds['hate_speech']),
    'toxicity': accuracy_score(all_labels['toxicity'], all_preds['toxicity' ])
    }
    
    return accuracies
# Split data into training and validation sets
train_df = merged_df.sample(frac=0.8, random_state=42)
val_df = merged_df.drop(train_df.index)
# Create datasets and dataloaders
max_len = 128
train_dataset = MultiTaskDataset (train_df, tokenizer, max_len) 
val_dataset = MultiTaskDataset(val_df, tokenizer, max_len)
best_model = None 
best_score = 0
# Hyperparameter tuning

for params in param_combinations:
    # Create dataloaders with current batch size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params[ 'batch_size'], shuffle=False)
    # Initialize model and optimizer with current Learning rate and weight decay
    model = MultiTaskModel (base_model)
    optimizer = optim. Adamb (model. parameters(), lr=params['learning_rate'], weight_decay=params['weight _decay'])
    # Training Loop
    for epoch in range (params ['num_epochs']) :
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_ mask']
            labels = batch['labels']
            optimizer.zero_grad()
            fake_news_logits, hate_speech_logits, toxicity_logits = model (input_ids, attention_mask)
            loss_fake_news = criterion(fake_news_logits, labels['fake_news'])
            loss_hate_speech = criterion(hate_speech_logits, labels[ 'hate_speech'])
            loss_toxicity = criterion(toxicity_logits, labels['toxicity'])
            total_loss = loss_fake_news + loss_hate_speech + loss_toxicity 
            total_loss.backward()
            optimizer.step()

    # Evaluate model
    scores = evaluate_model(model, val_loader)
    avg_score = sum(scores.values()) / len(scores)
    # Check if current model is the best
    if avg_score > best_score:
        best_score = avg_score
        best_model = model
        # Save the best model
        torch.save(best_model.state_dict(), 'multi_task_model.pth')
print (f"Best Score: {best_score}")
# Load the best model
model = MultiTaskModel (base_model)
model. load_state_dict(torch. load('multi_task_model.pth'))
model.eval()
# Function to make predictions on new text
def predict(text, model, tokenizer, max_len=128):
    inputs = tokenizer.encode_plus(
    text,
    None,
    add_special_tokens=True,
    max_length=max_len, 
    pad_to_max_length=True, 
    return_token_type_ids=False, 
    truncation=True
    )

    input_ids = torch. tensor (inputs [' input_ids'], dtype-torch.long).unsqueeze(0)
    attention_mask - torch. tensor (inputs['attention mask'], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        fake_news_logits, hate_speech_logits, toxicity_logits = model(input_ids, attention_mask)
        fake_news_pred = (torch.sigmoid(fake_news_logits) > 0.5).int().item()
        hate_speech_pred = (torch.sigmoid (hate_speech_logits) > 0.5).int().item() 
        toxicity_pred = (torch.sigmoid (toxicity_logits) > 0.5).int().item()
    return [fake_news_pred, hate_speech_pred, toxicity_pred]
# Example usage
text = "This is an example tweet."
predictions = predict(text, model, tokenizer)
print(f"Predictions: {predictions}") # Output: [0, 1, 1] or similar
