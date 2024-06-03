import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.utils
from torch.utils.data import dataset
import torch.utils.data
from transformer import TransformerModel
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
ntokens = 66058  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# Instantiate the Transformer model and set to eval
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

# Load model and vocabulary
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
vocab = checkpoint['vocab']

model.eval()

# Tokenize input text
tokenizer = get_tokenizer('basic_english')
seed_text = "the clock was"
tokenized_input = tokenizer(seed_text)

# Convert tokens to indices using vocabulary mapping
indexed_input = [vocab[token] for token in tokenized_input]

# Convert indices to tensor
input_tensor = torch.tensor(indexed_input)

# Initialize output text with seed text
output_text = seed_text

# Number of words to generate
num_words_to_generate = 5

# Top-k value (adjust as needed)
top_k = 4

# Forward pass
with torch.no_grad():
    for _ in range(num_words_to_generate):
        output = model(input_tensor.unsqueeze(1))  # Add batch dimension
        
        # Convert output logits/probabilities to token indices
        output_indices = output.argmax(dim=-1)
        
        # Get the top-k token indices
        top_k_indices = torch.topk(output[:, -1, :], top_k, dim=-1).indices
        
        # Sample a token index from the top-k indices
        last_predicted_index = top_k_indices[0][torch.randint(0, top_k, (1,))].item()
        
        # Get the corresponding token
        last_predicted_token = vocab.get_itos()[last_predicted_index]
        
        # Append the predicted token to the output text
        output_text += ' ' + last_predicted_token
        
        # Update the input tensor with the last predicted token
        input_tensor = torch.cat((input_tensor, torch.tensor([last_predicted_index])), dim=0)
    
    # Print the generated text
    print(output_text)