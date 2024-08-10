import onnxruntime
import numpy as np
import json
import argparse
from nltk import word_tokenize
from pathlib import Path

# Define the base directory as the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the ONNX model
onnx_model = f'{BASE_DIR}/model/model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model)

# Assuming pos_tag_to_idx and ner_tag_to_idx are saved in a JSON file
with open(f'{BASE_DIR}/model/tag_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)
    pos_tag_to_idx = mappings['pos_tag_to_idx']
    ner_tag_to_idx = mappings['ner_tag_to_idx']

# Inverse mappings
idx_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_idx.items()}
idx_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_idx.items()}

def preprocess_tokens(tokens, word_to_idx, max_length):
    """
    Preprocess the input tokens by converting them into their corresponding indices
    based on a word-to-index mapping and applying padding to match the max length.

    Args:
        tokens (list of str): List of input tokens (words).
        word_to_idx (dict): Mapping from words to their corresponding indices.
        max_length (int): Maximum length of the input sequence for padding.

    Returns:
        np.array: Array of token indices with applied padding.
    """
    token_indices = [word_to_idx.get(word, 0) for word in tokens]

    # Padding
    padding_length = max_length - len(token_indices)
    token_indices += [0] * padding_length

    return np.array(token_indices, dtype=np.int64)

# Load word_to_idx and max_length from the saved file or define them here
with open(f'{BASE_DIR}/model/model_params.json', 'r', encoding='utf-8') as f:
    model_params = json.load(f)
    word_to_idx = model_params['word_to_idx']
    max_length = model_params['max_length']

def predict(input_tokens):
    """
    Predict the POS and NER tags for a given input sentence.

    Args:
        input_tokens (list of str): List of tokens (words) from the input sentence.

    Returns:
        tuple: Two lists containing the predicted POS tags and NER tags, respectively.
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

    # Remove padding tokens and return predictions
    valid_length = len(input_tokens)
    pos_preds = pos_preds[0][:valid_length]
    ner_preds = ner_preds[0][:valid_length]

    # Convert indices back to tags
    predicted_pos_tags = [idx_to_pos_tag.get(idx, 'OTH') for idx in pos_preds]
    predicted_ner_tags = [idx_to_ner_tag.get(idx, 'OTH') for idx in ner_preds]

    return predicted_pos_tags, predicted_ner_tags

# Main function to handle input from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict POS and NER tags for a given input sentence.")
    parser.add_argument('--input', type=str, required=True, help='Input sentence for prediction.')
    
    args = parser.parse_args()
    input_sentence = args.input
    
    input_tokens = word_tokenize(input_sentence)
    pos_tags, ner_tags = predict(input_tokens)
    
    print("Input Tokens:", input_tokens)
    print("Predicted POS tags:", pos_tags)
    print("Predicted NER tags:", ner_tags)
