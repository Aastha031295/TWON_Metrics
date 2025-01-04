"""
Ref
`cltrier_lib` # official library
"""

import requests

from prompt import CLASSIFICATION_PROMPT

from config import USE_UNI_LLM_API, UNI_MODEL, TEMPERATURE


UNI_LLM_API: str = "https://inf.cl.uni-trier.de/chat/"


options = {"seed": 42, "temperature": TEMPERATURE, "num_predict": 128}


def predict_with_uni_llm(tweet):
    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": tweet},
    ]

    response = requests.post(
        UNI_LLM_API,
        json={"model": UNI_MODEL, "messages": messages, "options": options},
    ).json()
    return response["response"]


def predict(tweet):
    if USE_UNI_LLM_API:
        return predict_with_uni_llm(tweet)
    else:
        pass
        # return predict_with_huggingface(tweet)
