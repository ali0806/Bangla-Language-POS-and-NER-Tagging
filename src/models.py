import torch
import torch.nn as nn

class BLSTM_CNN(nn.Module):
    """
    A Bidirectional LSTM with CNN (BLSTM_CNN) model for sequence labeling tasks, 
    specifically for POS tagging and NER.

    Args:
        embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        lstm_units (int): Number of hidden units in the LSTM layer.
        pos_tag_size (int): Number of unique POS tags in the output.
        ner_tag_size (int): Number of unique NER tags in the output.
        char_embedding_size (int): Size of the character-level embeddings.
        cnn_out_channels (int): Number of output channels for the CNN layer.
        cnn_kernel_size (int): Kernel size for the CNN layer.
        dropout_rate (float): Dropout rate to apply after the LSTM layer.

    Methods:
        forward(x):
            Computes the forward pass of the model, returning logits for POS and NER tags.
    """
    
    def __init__(self, embedding_matrix, lstm_units, pos_tag_size, ner_tag_size, char_embedding_size, cnn_out_channels, cnn_kernel_size, dropout_rate):
        super(BLSTM_CNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.char_embedding = nn.Embedding(vocab_size, char_embedding_size)
        self.cnn = nn.Conv1d(char_embedding_size, cnn_out_channels, kernel_size=cnn_kernel_size, padding=1)
        self.lstm = nn.LSTM(embedding_dim + cnn_out_channels, lstm_units, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_pos = nn.Linear(lstm_units * 2, pos_tag_size)
        self.fc_ner = nn.Linear(lstm_units * 2, ner_tag_size)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            tuple: 
                - pos_logits (torch.Tensor): Logits for POS tags with shape (batch_size, sequence_length, pos_tag_size).
                - ner_logits (torch.Tensor): Logits for NER tags with shape (batch_size, sequence_length, ner_tag_size).
        """
        embedded = self.embedding(x)
        char_embedded = self.char_embedding(x).permute(0, 2, 1)
        char_cnn = self.cnn(char_embedded).permute(0, 2, 1)
        combined = torch.cat((embedded, char_cnn), dim=2)
        
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)

        pos_logits = self.fc_pos(lstm_out)
        ner_logits = self.fc_ner(lstm_out)
        
        return pos_logits, ner_logits
