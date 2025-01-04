
UNI_MODELS = [
    # LLama (MetaAI)
    "llama3.1:8b-instruct-q6_K", #0
    "llama3.1:70b-instruct-q6_K", #1
    "llama3.3:70b-instruct-q6_K", #2
    # Mi(s/x)tral (Mistral AI)
    "mistral:7b-instruct-v0.2-q6_K", #3
    "mixtral:8x7b-instruct-v0.1-q6_K", #4
    # Phi (Mircosoft)
    "phi3:14b-medium-128k-instruct-q6_K", #5
    "phi3.5:3.8b-mini-instruct-q6_K", #6
    # Gemma (Google)
    "gemma:7b-instruct-q6_K", #7
    "gemma2:27b-instruct-q6_K", #8
    # QWEN (Alibaba)
    "qwen2:72b-instruct-q6_K", #9
]

# Configuration
DATA_PATH = "balanced_dataset.csv" # Path to the dataset
SAMPLE_COUNT = 30 # Number of samples to evaluate

USE_UNI_LLM_API: bool = True # False to use Hugging Face API
UNI_MODEL:str = UNI_MODELS[9] # Change the index to select a different model
HG_MODEL: str = "facebook/bart-large" # Hugging Face model to use
TEMPERATURE: float = 0.3 # Value between 0.0 and 1.0 to control the randomness of the predictions
