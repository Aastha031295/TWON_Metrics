import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Step 1: Define a Dataset class for loading data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 2: Load and preprocess data
def load_data(file_path, text_column, label_column, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    return train_texts, val_texts, train_labels, val_labels

# Step 3: Initialize tokenizer and model
def initialize_model_and_tokenizer(model_name, num_labels):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# Step 4: Adjust batch size dynamically based on GPU memory
def get_dynamic_batch_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - reserved_memory
        # Estimate batch size based on free memory (heuristic)
        return min(32, max(8, free_memory // (128 * 1024**2)))
    return 16  # Default batch size for CPUs

# Step 5: Training pipeline
def train_model(train_texts, val_texts, train_labels, val_labels, tokenizer, model, max_length, batch_size, epochs, output_dir):
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    return trainer

# Step 6: Main script
if __name__ == "__main__":
    # Configurations
    DATA_FILE = "data.csv"  # Path to your dataset
    TEXT_COLUMN = "text"  # Column name containing the text
    LABEL_COLUMN = "label"  # Column name containing the labels
    MODEL_NAME = "bert-base-uncased"  # Pretrained model name
    NUM_LABELS = 2  # Number of classes
    MAX_LENGTH = 128  # Max length of input text
    EPOCHS = 3
    OUTPUT_DIR = "bert_finetuned"

    # Dynamically determine batch size
    BATCH_SIZE = get_dynamic_batch_size()

    # Load and preprocess data
    train_texts, val_texts, train_labels, val_labels = load_data(DATA_FILE, TEXT_COLUMN, LABEL_COLUMN)

    # Initialize model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(MODEL_NAME, NUM_LABELS)

    # Train model
    trainer = train_model(train_texts, val_texts, train_labels, val_labels, tokenizer, model, MAX_LENGTH, BATCH_SIZE, EPOCHS, OUTPUT_DIR)

    print("Training completed. Model saved to", OUTPUT_DIR)
