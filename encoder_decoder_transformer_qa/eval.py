import math
import torch
import warnings
from torchtext.data.utils import get_tokenizer
from transformer import EncoderDecoderTransformer
import torch.nn.functional as F

# Suppress specific torchtext deprecation warning
warnings.filterwarnings("ignore", message="torchtext is deprecated and will be removed in a future release")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
ntokens = 66058  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = EncoderDecoderTransformer(
    src_vocab_size=ntokens,
    tgt_vocab_size=ntokens,
    d_model=emsize,
    nhead=nhead,
    d_hid=d_hid,
    enc_layers=nlayers,
    dec_layers=nlayers,
    dropout=dropout
).to(device)

# Load model and vocabulary
checkpoint = torch.load('model_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
vocab = checkpoint['vocab']

# Set model to evaluation mode
model.eval()

# Tokenize input text
tokenizer = get_tokenizer('basic_english')
input_text = "the information is part of the"
tokenized_input = tokenizer(input_text)

# Ensure tokenized_input is a list of tokens
if isinstance(tokenized_input, str):
    tokenized_input = [tokenized_input]

# Check tokens and vocabulary
print("Tokenized input:", tokenized_input)

# Get the End-of-sequence token ID
eos_token_id = vocab['<eos>']  # End-of-sequence token ID (replace '<eos>' with your actual EOS token)

# Convert tokens to indices using vocabulary mapping
indexed_input = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokenized_input]

# Check the indexed input
print("Indexed input:", indexed_input)

# Convert indices to tensor
input_tensor = torch.tensor(indexed_input).unsqueeze(1).to(device)  # Add batch dimension

# Initialize the target sequence with the start-of-sequence token
tgt_start_token_id = vocab['<sos>']  # Start-of-sequence token ID (replace '<sos>' with your actual SOS token)
tgt_indices = [tgt_start_token_id]
output_tokens = []

# Temperature for temperature scaling (adjust as needed)
temperature = 1  # Adjust this value to control the diversity of generated samples

# Top-k value for top-k sampling (adjust as needed)
top_k = 20  # Adjust this value to control the diversity of generated samples

max_len = 10

with torch.no_grad():
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(1).to(device)  # Add batch dimension

        # Calculate the sequence length
        seq_length = tgt_tensor.size(0)

        # Create the target mask
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length)).transpose(0, 1).type_as(tgt_tensor)
        tgt_mask = (tgt_mask == 0).unsqueeze(0)

        # Create source mask (set to None for the sake of this example)
        src_mask = None

        # Forward pass
        output = model(input_tensor, tgt_tensor, src_mask)

        # Get the token probabilities using temperature scaling
        token_probs = F.softmax(output[-1, :, :] / temperature, dim=1)
        #print("Token Probabilities:", token_probs)

        # Apply top-k sampling to get the candidate tokens
        topk_probs, topk_indices = token_probs.topk(top_k, dim=1)

        # Sample from the top-k indices
        next_token_id = topk_indices[0, torch.multinomial(topk_probs[0], 1).item()].item()

        # Append the generated token to the target sequence
        tgt_indices.append(next_token_id)
        output_tokens.append(next_token_id)

        # Stop if the end-of-sequence token is generated
        if next_token_id == eos_token_id:
            break

# Convert generated token indices back to tokens
reverse_vocab = {idx: token for token, idx in vocab.get_stoi().items()}  # Get index-to-string mapping
generated_tokens = [reverse_vocab.get(idx, '<unk>') for idx in output_tokens]

# Join tokens to form the final generated text
generated_text = ' '.join(generated_tokens)
print("Generated Text:", generated_text)