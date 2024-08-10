import torch
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

# Define the base directory as the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent.parent

def train_model(model, data_loader, optimizer, criterion, device):
    """
    Train the model for one epoch on the provided data loader.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function used for training.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        float: The average loss over all batches in the data loader.
    """
    model.train()
    total_loss = 0

    for batch in data_loader:
        tokens, pos_labels, ner_labels = zip(*batch)
        tokens = torch.stack(tokens).to(device)
        pos_labels = torch.stack(pos_labels).to(device)
        ner_labels = torch.stack(ner_labels).to(device)
        
        optimizer.zero_grad()
        pos_logits, ner_logits = model(tokens)
        
        loss_pos = criterion(pos_logits.view(-1, pos_logits.shape[-1]), pos_labels.view(-1))
        loss_ner = criterion(ner_logits.view(-1, ner_logits.shape[-1]), ner_labels.view(-1))
        loss = loss_pos + loss_ner
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the provided validation/test data loader.

    Args:
        model (nn.Module): The model to be evaluated.
        data_loader (DataLoader): DataLoader providing the evaluation data.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        dict: A dictionary containing the precision, recall, f1-score, and accuracy for both POS and NER tagging.
    """
    model.eval()
    all_pos_preds = []
    all_ner_preds = []
    all_pos_labels = []
    all_ner_labels = []

    with torch.no_grad():
        for batch in data_loader:
            tokens, pos_labels, ner_labels = zip(*batch)
            tokens = torch.stack(tokens).to(device)
            pos_labels = torch.stack(pos_labels).to(device)
            ner_labels = torch.stack(ner_labels).to(device)
            
            pos_logits, ner_logits = model(tokens)
            
            pos_preds = torch.argmax(pos_logits, dim=-1).cpu().numpy()
            ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            pos_labels = pos_labels.cpu().numpy()
            ner_labels = ner_labels.cpu().numpy()

            all_pos_preds.append(pos_preds)
            all_ner_preds.append(ner_preds)
            all_pos_labels.append(pos_labels)
            all_ner_labels.append(ner_labels)

    pos_preds = np.concatenate([p.flatten() for p in all_pos_preds])
    ner_preds = np.concatenate([n.flatten() for n in all_ner_preds])
    pos_labels = np.concatenate([l.flatten() for l in all_pos_labels])
    ner_labels = np.concatenate([l.flatten() for l in all_ner_labels])

    # Filter out padding tokens (assumed to be labeled with 0)
    mask = pos_labels != 0
    pos_preds = pos_preds[mask]
    pos_labels = pos_labels[mask]

    mask = ner_labels != 0
    ner_preds = ner_preds[mask]
    ner_labels = ner_labels[mask]

    pos_report = classification_report(pos_labels, pos_preds, output_dict=True)
    ner_report = classification_report(ner_labels, ner_preds, output_dict=True)
    
    return {
        'pos': {
            'precision': pos_report['macro avg']['precision'],
            'recall': pos_report['macro avg']['recall'],
            'f1-score': pos_report['macro avg']['f1-score']
        },
        'ner': {
            'precision': ner_report['macro avg']['precision'],
            'recall': ner_report['macro avg']['recall'],
            'f1-score': ner_report['macro avg']['f1-score']
        },
        'accuracy_ner': ner_report['accuracy'],
        'accuracy_pos': pos_report['accuracy'],
    }

def save_model_as_onnx(model, word_vectors, max_length, device):
    """
    Save the trained model in ONNX format for deployment or inference.

    Args:
        model (nn.Module): The trained model to be saved.
        word_vectors (dict): The word vectors used in the model.
        max_length (int): The maximum sequence length for input data.
        device (torch.device): Device on which to perform computations (CPU or GPU).
    """
    dummy_input = torch.randint(0, len(word_vectors), (1, max_length)).to(device)
    onnx_path = f"{BASE_DIR}/model/model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=["input"], 
                      output_names=["pos_output", "ner_output"],
                      dynamic_axes={"input": {0: "batch_size"}, "pos_output": {0: "batch_size"}, "ner_output": {0: "batch_size"}},
                      opset_version=11)
    print(f"Model saved in ONNX format at {onnx_path}")
