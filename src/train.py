import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from data_utils import load_data, prepare_word_vectors, generate_tag_mappings, CustomDataset
from models import BLSTM_CNN
from utils import train_model, evaluate_model, save_model_as_onnx
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a BLSTM-CNN model for POS and NER tagging.')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization).')
parser.add_argument('--onnx_model_save', type=bool, default=True, help='Save model as ONNX format (True/False).')

args = parser.parse_args()

# Static values for remaining parameters
lstm_units = 128
char_embedding_size = 50
cnn_out_channels = 30
cnn_kernel_size = 3


# Hyperparameters and settings from arguments
num_epochs = args.num_epochs
batch_size = args.batch_size
dropout_rate = args.dropout_rate
learning_rate = args.learning_rate
weight_decay = args.weight_decay
onnx_model_save = args.onnx_model_save

# Load data and prepare word vectors
filepath = f'{BASE_DIR}/data/processed/processed_data.json'
data = load_data(filepath)
embedding_matrix, word_vectors = prepare_word_vectors(data)
pos_tag_to_idx, ner_tag_to_idx = generate_tag_mappings(data)
max_length = max(len(item['tokens']) for item in data)

# Split data into train, val, and test sets
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - train_size - val_size

train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])

# Create datasets and data loaders
train_dataset = CustomDataset(train_data, word_vectors, pos_tag_to_idx, ner_tag_to_idx, max_length)
val_dataset = CustomDataset(val_data, word_vectors, pos_tag_to_idx, ner_tag_to_idx, max_length)
test_dataset = CustomDataset(test_data, word_vectors, pos_tag_to_idx, ner_tag_to_idx, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

# Set device, model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BLSTM_CNN(embedding_matrix, lstm_units, len(pos_tag_to_idx) + 1, len(ner_tag_to_idx) + 1, char_embedding_size, cnn_out_channels, cnn_kernel_size, dropout_rate).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Training loop with early stopping
best_val_loss = float('inf')
patience = 5
trigger_times = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    """
    Main training loop for the BLSTM-CNN model. For each epoch, the model is trained on the training dataset,
    evaluated on the validation dataset, and if a lower validation loss is achieved, the model is saved.
    
    Implements early stopping to prevent overfitting by halting training if validation loss does not improve 
    for a specified number of epochs.
    """
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate_model(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_metrics)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Metrics: POS - {val_metrics['pos']}, NER - {val_metrics['ner']}")

    if train_loss < best_val_loss:
        best_val_loss = train_loss
        trigger_times = 0
        torch.save(model.state_dict(), f'{BASE_DIR}/model/best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Evaluate on test data
model.load_state_dict(torch.load(f'{BASE_DIR}/model/best_model.pth'))
test_metrics = evaluate_model(model, test_loader, criterion, device)
print(f"Test Metrics: POS - {test_metrics['pos']}, NER - {test_metrics['ner']}")
print(f"Test POS Accuracy: {test_metrics['accuracy_pos']}")
print(f"Test NER Accuracy: {test_metrics['accuracy_ner']}")

# Save the model in ONNX format if specified
if onnx_model_save:
    save_model_as_onnx(model, word_vectors, max_length, device)

# Save tag mappings
mappings = {
    "pos_tag_to_idx": pos_tag_to_idx,
    "ner_tag_to_idx": ner_tag_to_idx
}
with open(f'{BASE_DIR}/model/tag_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(mappings, f, ensure_ascii=False)

# Save model parameters
model_params = {
    "word_to_idx": train_dataset.word_to_idx,
    "max_length": max_length
}
with open(f'{BASE_DIR}/model/model_params.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False)

# Plot and save training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.savefig(f'{BASE_DIR}/media/training_loss_curve.png')
