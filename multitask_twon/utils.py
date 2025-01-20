import torch
from transformers import BertTokenizer
from model import MultiTaskModel

def load_model(model_path, device):
    """
    Load the trained multi-task model.
    """
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_tokenizer():
    """
    Load the tokenizer for the BERT-based model.
    """
    return BertTokenizer.from_pretrained("bert-base-german-cased")

import os
import pandas as pd

def save_predictions(df, output_path, file_name="predictions.csv"):
    """
    Save predictions to a CSV file.
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, file_name)
    df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

