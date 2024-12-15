import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(pred):
    """Compute evaluation metrics like accuracy and F1."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the dataset with truncation and padding."""
    return tokenizer(
        examples['text'],  # Change this column name based on your dataset
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def prepare_dataset(dataset_name, tokenizer, max_length):
    """Load and preprocess the dataset."""
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Tokenize dataset
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length), 
        batched=True
    )

    # Set format for PyTorch or TensorFlow
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {'labels': x['label']}, batched=True
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_datasets

def train_model(model_name, dataset_name, output_dir, num_labels, max_length, num_epochs):
    """Train and fine-tune the model on a given dataset."""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Prepare dataset
    datasets = prepare_dataset(dataset_name, tokenizer, max_length)
    train_dataset = datasets['train']
    eval_dataset = datasets['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="epoch",
        save_total_limit=2,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    # Save the final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return results

# Example Usage
if __name__ == "__main__":
    # Define your parameters
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = {
        "model_name": "bert-base-uncased",
        "dataset_name": "imdb",  # Replace with your dataset
        "output_dir": "./fine_tuned_model",
        "num_labels": 2,  # Adjust based on your task
        "max_length": 128,
        "num_epochs": 3,
    }

    # Train the model
    results = train_model(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        output_dir=config["output_dir"],
        num_labels=config["num_labels"],
        max_length=config["max_length"],
        num_epochs=config["num_epochs"],
    )

    # Print final results
    print("Evaluation Results:", results)
