
 
 
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os
 
import gc
torch.cuda.empty_cache()
gc.collect()
# Set the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
 
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
 
# Function to load the model
def load_model():
    model = MultiTaskModel().to(device)  # Initialize the model
    model.load_state_dict(torch.load('../../media/data/multiTaskTWONB1/multi_task_model.pt'))  # Load the saved state
    model.eval()  # Set the model to evaluation mode
    return model
 
# Function to make predictions
def predict(text, model):
    # Tokenize and encode the input text
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
   
    # Move input tensors to the same device as the model
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
   
    # Make predictions
    with torch.no_grad():
        outputs_fake_news, outputs_hate_speech, outputs_toxicity = model(input_ids, attention_mask)
       
    # Apply sigmoid to get probabilities and round to get binary predictions
    preds_fake_news = torch.sigmoid(outputs_fake_news).squeeze().round().cpu().numpy()
    preds_hate_speech = torch.sigmoid(outputs_hate_speech).squeeze().round().cpu().numpy()
    preds_toxicity = torch.sigmoid(outputs_toxicity).squeeze().round().cpu().numpy()
   
    return preds_fake_news, preds_hate_speech, preds_toxicity
 
# Load the model
model = load_model()
 
# Example text input for prediction
text_input = "Mir fallen nur Steuervorteile durch Gender Pay gap ein."
 
# Make predictions
predictions = predict(text_input, model)
 
# Print the predictions
print(f"Fake News Prediction: {predictions[0]}")
print(f"Hate Speech Prediction: {predictions[1]}")
print(f"Toxicity Prediction: {predictions[2]}")


torch.cuda.empty_cache()
gc.collect()