import os
from pathlib import Path

LLM_PATH = " "
LLM_URL = ' '
LLM_PARAMS = {
    "max_tokens": 12000,
    "temperature": 0,
    "top_k": 1,
    "top_p": 0.8,
    "repetition_penalty": 1,
    "stream": False
}
HEADERS = {'Content-Type': 'application/json'}

MODEL_NAME = "BAAI/bge-large-zh-v1.5"