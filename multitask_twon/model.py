import torch.nn as nn
from transformers import BertModel

class MultiTaskModel(nn.Module):
    """
    Multi-task classification model for fake news, hate speech, and toxicity detection.
    """
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-german-cased")
        self.dropout = nn.Dropout(0.3)
        self.fc_fake_news = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_hate_speech = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_toxicity = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Extract pooled output
        pooled_output = self.dropout(pooled_output)
        fake_news = self.fc_fake_news(pooled_output)
        hate_speech = self.fc_hate_speech(pooled_output)
        toxicity = self.fc_toxicity(pooled_output)
        return fake_news, hate_speech, toxicity
