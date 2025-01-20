"""
Ref
`cltrier_lib` # official library
"""

import requests

from config import TEMPERATURE, UNI_MODEL, USE_UNI_LLM_API
from prompt import CLASSIFICATION_PROMPT

UNI_LLM_API: str = "https://inf.cl.uni-trier.de/chat/"


options = {"seed": 42, "temperature": TEMPERATURE, "num_predict": 128}


def predict_with_uni_llm(tweet):
    try:
        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": tweet},
        ]

        response = requests.post(
            UNI_LLM_API,
            json={"model": UNI_MODEL, "messages": messages, "options": options},
        )
        response_json = response.json()
        return response_json["response"]
    except Exception as exc:
        print(response)
        print(f"Error: {repr(exc)}")
        raise exc


def predict(tweet):
    if USE_UNI_LLM_API:
        return predict_with_uni_llm(tweet)
    else:
        raise NotImplementedError("HuggingFace not implemented yet.")
        # return predict_with_huggingface(tweet)