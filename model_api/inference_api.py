import onnxruntime
import numpy as np
import json
from nltk import word_tokenize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import nltk
from pathlib import Path

nltk.download('punkt')
nltk.download('punkt_tab')

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the ONNX model
onnx_model = f'{BASE_DIR}/model/model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model)

# Load the mappings from the saved file or define them here
with open(f'{BASE_DIR}/model/tag_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)
    pos_tag_to_idx = mappings['pos_tag_to_idx']
    ner_tag_to_idx = mappings['ner_tag_to_idx']

# Inverse mappings
idx_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_idx.items()}
idx_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_idx.items()}

# Load word_to_idx and max_length from the saved file or define them here
with open(f'{BASE_DIR}/model/model_params.json', 'r', encoding='utf-8') as f:
    model_params = json.load(f)
    word_to_idx = model_params['word_to_idx']
    max_length = model_params['max_length']

# FastAPI application
app = FastAPI()

# Request model
class TextRequest(BaseModel):
    text: str

# Function to preprocess input tokens
def preprocess_tokens(tokens, word_to_idx, max_length):
    """
    Preprocesses input tokens by converting them into indices using word_to_idx mapping
    and padding the sequence to max_length.

    Args:
        tokens (list of str): List of input tokens (words).
        word_to_idx (dict): Dictionary mapping words to their corresponding indices.
        max_length (int): Maximum length of the token sequence.

    Returns:
        np.ndarray: Array of token indices with padding if necessary.
    """
    token_indices = [word_to_idx.get(word, 0) for word in tokens]

    # Padding
    padding_length = max_length - len(token_indices)
    token_indices += [0] * padding_length

    return np.array(token_indices, dtype=np.int64)

# Define a function to predict POS and NER tags for a given input
def predict(input_tokens):
    """
    Predicts POS and NER tags for a given input sequence of tokens.

    Args:
        input_tokens (list of str): List of input tokens (words).

    Returns:
        tuple: Two lists containing the predicted POS tags and NER tags respectively.
    """
    # Preprocess the input tokens
    token_indices = preprocess_tokens(input_tokens, word_to_idx, max_length)
    token_indices = token_indices.reshape(1, -1)  # Add batch dimension

    # Make predictions using the ONNX model
    ort_inputs = {'input': token_indices}
    ort_outs = ort_session.run(None, ort_inputs)

    # Get the predicted POS and NER tags
    pos_preds = np.argmax(ort_outs[0], axis=-1)
    ner_preds = np.argmax(ort_outs[1], axis=-1)

    # Remove padding tokens
    valid_length = len(input_tokens)
    pos_preds = pos_preds[0][:valid_length]
    ner_preds = ner_preds[0][:valid_length]

    # Convert indices back to tags
    predicted_pos_tags = [idx_to_pos_tag.get(idx, 'OTH') for idx in pos_preds]
    predicted_ner_tags = [idx_to_ner_tag.get(idx, 'B-OTH') for idx in ner_preds]

    return predicted_pos_tags, predicted_ner_tags

@app.post("/predict")
def predict_tags(request: TextRequest):
    """
    FastAPI endpoint to predict POS and NER tags for the provided text input.

    Args:
        request (TextRequest): JSON request containing the input text.

    Returns:
        dict: A dictionary containing the input tokens, predicted POS tags, and predicted NER tags.
    """
    input_text = request.text
    if not input_text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    input_tokens = word_tokenize(input_text)
    pos_tags, ner_tags = predict(input_tokens)

    return {
        "input_tokens": input_text,
        "predicted_pos_tags": pos_tags,
        "predicted_ner_tags": ner_tags
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
