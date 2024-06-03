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

debug = True

###########################################################
###                                                     ###
###   LOAD DATA                                         ###    
###                                                     ###
###########################################################

# Load the SQuAD dataset
dataset = load_dataset('squad')
train_data = dataset['train']
val_data = dataset['validation']

###########################################################
###                                                     ###
###   BUILD VOCABULARY                                  ###    
###                                                     ###
###########################################################

tokenizer = get_tokenizer('basic_english')

# Function to yield tokens from the text
def yield_tokens(data_iter):
    for data in data_iter:
        yield tokenizer(data['context'])
        yield tokenizer(data['question'])

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

###########################################################
###                                                     ###
###   TOKENISATION                                      ###    
###                                                     ###
###########################################################

def data_process(data_iter):
    inputs = []
    targets = []
    for item in data_iter:
        context = torch.tensor(vocab(tokenizer(item['context'])), dtype=torch.long)
        question = torch.tensor(vocab(tokenizer(item['question'])), dtype=torch.long)
        input = torch.cat((context, question), dim=0)
        start_pos = item['answers']['answer_start'][0]
        end_pos = start_pos + len(item['answers']['text'][0])
        target = torch.tensor([start_pos, end_pos], dtype=torch.long)
        inputs.append(input)
        targets.append(target)
    return inputs, targets

train_inputs, train_targets = data_process(train_data)
val_inputs, val_targets = data_process(val_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################
###                                                     ###
###   DATA PRE-PROCESSING                               ###    
###                                                     ###
###########################################################

def batchify(data_list, target_list, bsz: int):
    num_batches = len(data_list) // bsz
    data_batches = []
    target_batches = []
    for i in range(num_batches):
        data_batch = data_list[i*bsz:(i+1)*bsz]
        target_batch = target_list[i*bsz:(i+1)*bsz]
        data_lengths = [len(data) for data in data_batch]
        max_length = max(data_lengths)
        padded_data = [torch.cat((data, torch.zeros(max_length - len(data), dtype=torch.long))) for data in data_batch]
        padded_targets = [torch.cat((target, torch.zeros(2 - len(target), dtype=torch.long))) for target in target_batch]
        data_batches.append(torch.stack(padded_data).t().contiguous().to(device))
        target_batches.append(torch.stack(padded_targets).t().contiguous().to(device))
    return data_batches, target_batches

batch_size = 30
eval_batch_size = 30
train_data_batches, train_target_batches = batchify(train_inputs, train_targets, batch_size)
val_data_batches, val_target_batches = batchify(val_inputs, val_targets, eval_batch_size)

bptt = 32  # sequence length cannot exceed bptt
def get_batch(source: Tensor, targets: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i+seq_len]
    target = targets[i:i+seq_len]
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
model = EncoderDecoderTransformer(src_vocab_size=ntokens, tgt_vocab_size=ntokens, d_model=emsize, nhead=nhead, d_hid=d_hid, enc_layers=nlayers, dec_layers=nlayers, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data_batches)
    for batch, (src, tgt) in enumerate(zip(train_data_batches, train_target_batches)):
        optimizer.zero_grad()
        output = model(src, tgt)
        output_flat = output.view(-1, ntokens)
        tgt_flat = tgt.contiguous().view(-1)
        loss = criterion(output_flat, tgt_flat)
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

def evaluate(model: nn.Module, data_batches, target_batches) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for src, tgt in zip(data_batches, target_batches):
            output = model(src, tgt)
            output_flat = output.view(-1, ntokens)
            tgt_flat = tgt.contiguous().view(-1)
            total_loss += criterion(output_flat, tgt_flat).item()
    return total_loss / len(data_batches)

best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data_batches, val_target_batches)
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
    model.load_state_dict(torch.load(best_model_params_path))  # load best model states
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, 'model_checkpoint.pth')

test_loss = evaluate(model, val_data_batches, val_target_batches)  # For simplicity, using val_data as test_data
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
