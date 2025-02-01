import pandas as pd
from torch.utils.data import Dataset

class SocialMediaDataset(Dataset):
    """
    Dataset class for tokenizing and batching the text data.
    """
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe["text"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": text
        }

def load_dataset(file_path):
    """
    Load dataset and check for correctness.
    """
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {len(df)} rows")
    print(df.head())  # Show some sample rows
    return df
