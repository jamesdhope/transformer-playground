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
from transformer import EncoderDecoderTransformer
from datasets import load_dataset
import time
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()

debug=True

###########################################################
###                                                     ###
###   LOAD DATA                                         ###    
###                                                     ###
###########################################################

# Load the WikiText2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_data = dataset['train']

###########################################################
###                                                     ###
###   BUILD VOCABULARY                                  ###    
###                                                     ###
###########################################################

tokenizer = get_tokenizer('basic_english')

# Function to yield tokens from the text
def yield_tokens(data_iter):
    for data in data_iter:
        yield tokenizer(data['text'])

''' The size of the embedding layer is determined by the vocabulary size (vocab_size) and the embedding dimension (embed_dim) hence we need to build a vocab to calculate the size'''
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

###########################################################
###                                                     ###
###   TOKENISATION                                      ###    
###                                                     ###
###########################################################

''' this function processes raw text data by tokenizing it, converting tokens to indices using a vocabulary, and then concatenating the resulting tensors into a single flat tensor. The output tensor contains the indices representing the tokens in the original raw text data.'''
def data_process(raw_text_iter: torch.utils.data.dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item['text'])), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter = dataset['train']
test_iter = dataset['test']
val_iter = dataset['validation']

train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################
###                                                     ###
###   DATA PRE-PROCESSING                               ###    
###                                                     ###
###########################################################

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 30
eval_batch_size = 30
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 32 # sequence length cannot exceed bptt
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Desc:
        In the get_batch function, src is assigned the current sequence from i to i+seq_len. However, tgt is assigned the next sequence, starting from i+1 to i+1+seq_len, effectively shifting the target sequence by one token ahead compared to the source sequence. This is because in language modeling tasks, the target for each token is the next token in the sequence.
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    
    # shifts one sequence ahead
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = torch.full_like(data,0) # target is padded to same length as source which means we no longer have to reshape it later on.
    target = source[i+1:i+1+seq_len] 
    return data, target

###########################################################
###                                                     ###
###   MODEL TRAINING                                    ###    
###                                                     ###
###########################################################

ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = EncoderDecoderTransformer(src_vocab_size=ntokens, tgt_vocab_size=ntokens, d_model=emsize, nhead=nhead, d_hid=d_hid, enc_layers=nlayers,dec_layers=nlayers, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.05  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        src, tgt = get_batch(train_data, i)

        # Forward pass
        output = model(src, tgt)

        # Reshape output and targets for loss calculation
        output_flat = output.view(-1, ntokens)
        tgt_flat = tgt.contiguous().view(-1)

        # Calculate loss
        loss = criterion(output_flat, tgt_flat)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            src, tgt = get_batch(eval_data, i)
            
            output = model(src, tgt)  # Pass both source and target sequences
            output_flat = output.view(-1, ntokens)
            tgt_flat = tgt.contiguous().view(-1)
            total_loss += criterion(output_flat, tgt_flat).item()
    return total_loss / (len(eval_data) - 1)

best_val_loss = float('inf')
epochs = 6

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    #torch.save(model.state_dict(), 'best_transformer_weights.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, 'model_checkpoint.pth')

test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
