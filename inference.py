"""
Ref
`cltrier_lib` # official library
"""

import requests

from prompt import CLASSIFICATION_PROMPT

UNI_LLM_API: str = "https://inf.cl.uni-trier.de/chat/"
TEMPERATURE = 0.2  # deterministic output

MODELS = [
    # LLama (MetaAI)
    "llama3.1:8b-instruct-q6_K",
    "llama3.1:70b-instruct-q6_K",
    "llama3.3:70b-instruct-q6_K",
    # Mi(s/x)tral (Mistral AI)
    "mistral:7b-instruct-v0.2-q6_K",
    "mixtral:8x7b-instruct-v0.1-q6_K",
    # Phi (Mircosoft)
    "phi3:14b-medium-128k-instruct-q6_K",
    "phi3.5:3.8b-mini-instruct-q6_K",
    # Gemma (Google)
    "gemma:7b-instruct-q6_K",
    "gemma2:27b-instruct-q6_K",
    # QWEN (Alibaba)
    "qwen2:72b-instruct-q6_K",
]

options = {"seed": 42, "temperature": TEMPERATURE, "num_predict": 128}


def predict(tweet, model=MODELS[0]):
    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": tweet},
    ]

    response = requests.post(
        UNI_LLM_API,
        json={"model": model, "messages": messages, "options": options},
    ).json()
    return response["response"]
