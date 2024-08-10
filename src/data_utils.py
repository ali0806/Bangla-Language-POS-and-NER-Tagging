import torch
from torch.utils.data import Dataset
import numpy as np
import json
from bnlp import BengaliWord2Vec
from pathlib import Path

# Define the base directory as the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(filepath):
    """
    Load tokenized text data from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing the data.

    Returns:
        list: A list of dictionaries where each dictionary represents a sentence 
              with 'tokens', 'pos_tag', and 'ner_tags'.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def prepare_word_vectors(data):
    """
    Prepare word embeddings using BengaliWord2Vec for the tokens in the dataset.

    Args:
        data (list): A list of dictionaries, where each dictionary contains 'tokens' 
                     as a key for the tokenized sentence.

    Returns:
        tuple: 
            - embedding_matrix (np.ndarray): A matrix where each row corresponds to a word vector.
            - word_vectors (dict): A dictionary mapping words to their corresponding word vectors.
    """
    bwv = BengaliWord2Vec()
    word_vectors = {}
    vector_size = None

    for item in data:
        for word in item['tokens']:
            try:
                word_vectors[word] = bwv.get_word_vector(word)
                if vector_size is None:
                    vector_size = word_vectors[word].shape[0]
            except KeyError:
                word_vectors[word] = np.zeros(vector_size or 300)
    
    vector_size = vector_size or 300
    embedding_matrix = np.zeros((len(word_vectors) + 1, vector_size))
    for i, (word, vec) in enumerate(word_vectors.items(), 1):
        embedding_matrix[i] = vec

    return embedding_matrix, word_vectors

def generate_tag_mappings(data):
    """
    Generate mappings from POS and NER tags to unique indices.

    Args:
        data (list): A list of dictionaries, where each dictionary contains 'pos_tag' 
                     and 'ner_tags' as keys for the respective tags.

    Returns:
        tuple:
            - pos_tag_to_idx (dict): A dictionary mapping POS tags to unique indices.
            - ner_tag_to_idx (dict): A dictionary mapping NER tags to unique indices.
    """
    pos_tag_set = set()
    ner_tag_set = set()

    for item in data:
        pos_tag_set.update(item['pos_tag'])
        ner_tag_set.update(item['ner_tags'])

    pos_tag_to_idx = {tag: idx for idx, tag in enumerate(pos_tag_set, 1)}
    ner_tag_to_idx = {tag: idx for idx, tag in enumerate(ner_tag_set, 1)}

    return pos_tag_to_idx, ner_tag_to_idx

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling token, POS, and NER data.

    Args:
        data (list): A list of dictionaries where each dictionary contains 'tokens', 
                     'pos_tag', and 'ner_tags'.
        word_vectors (dict): A dictionary mapping words to their corresponding word vectors.
        pos_tag_to_idx (dict): A dictionary mapping POS tags to unique indices.
        ner_tag_to_idx (dict): A dictionary mapping NER tags to unique indices.
        max_length (int): The maximum length of the sequences. Sequences shorter than this 
                          length will be padded with zeros.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves a sample from the dataset by index and returns the tokens, 
                          POS labels, and NER labels as tensors.
    """
    
    def __init__(self, data, word_vectors, pos_tag_to_idx, ner_tag_to_idx, max_length):
        """
        Initialize the CustomDataset with the provided data, word vectors, tag mappings, 
        and maximum sequence length.

        Args:
            data (list): A list of tokenized sentences with POS and NER tags.
            word_vectors (dict): A dictionary mapping words to their corresponding vectors.
            pos_tag_to_idx (dict): Mapping of POS tags to indices.
            ner_tag_to_idx (dict): Mapping of NER tags to indices.
            max_length (int): The maximum length of the sequences.
        """
        self.data = data
        self.word_vectors = word_vectors
        self.pos_tag_to_idx = pos_tag_to_idx
        self.ner_tag_to_idx = ner_tag_to_idx
        self.word_to_idx = {word: idx for idx, word in enumerate(word_vectors.keys(), 1)}
        self.max_length = max_length

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset by index and return the tokens, POS labels, 
        and NER labels as tensors.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: 
                - tokens (torch.LongTensor): A tensor of token indices.
                - pos_labels (torch.LongTensor): A tensor of POS tag indices.
                - ner_labels (torch.LongTensor): A tensor of NER tag indices.
        """
        item = self.data[idx]
        tokens = [self.word_to_idx.get(word, 0) for word in item['tokens']]
        pos_labels = [self.pos_tag_to_idx.get(tag, 0) for tag in item['pos_tag']]
        ner_labels = [self.ner_tag_to_idx.get(tag, 0) for tag in item['ner_tags']]
        
        padding_length = self.max_length - len(tokens)
        tokens += [0] * padding_length
        pos_labels += [0] * padding_length
        ner_labels += [0] * padding_length

        return torch.LongTensor(tokens), torch.LongTensor(pos_labels), torch.LongTensor(ner_labels)
