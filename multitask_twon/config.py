import torch

# Paths
MODEL_PATH = "multi_task_model.pt"  # Path to the trained model file
DATA_PATH = "merged_dataset_18jan.csv"  # Path to the dataset
OUTPUT_PATH = "output"  # Directory to save predictions

# Inference Settings
BATCH_SIZE = 32  # Adjust based on hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
MAX_LEN = 128  # Maximum token length
