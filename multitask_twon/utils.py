import os
import torch
from transformers import BertTokenizer
from model import MultiTaskModel
from config import MODEL_PATH

def load_model(device):
    """
    Load the trained multi-task model onto the specified device (CPU or GPU).
    """
    model = MultiTaskModel()  # Initialize your model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load the model weights from the specified path
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode
    return model

def load_tokenizer():
    """
    Load the tokenizer for the BERT-based model.
    """
    return BertTokenizer.from_pretrained("bert-base-german-cased")


def save_predictions(df, output_path, file_name="predictions.csv"):
    """
    Save predictions to a CSV file.
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, file_name)
    df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

