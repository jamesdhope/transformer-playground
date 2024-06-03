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
import time

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()

import torchtext
torchtext.disable_torchtext_deprecation_warning()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderDecoderTransformer(nn.Module):

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 enc_layers: int, dec_layers: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = TransformerEncoderModel(src_vocab_size, d_model, nhead, d_hid, enc_layers, dropout)
        self.decoder = TransformerDecoderModel(tgt_vocab_size, d_model, nhead, d_hid, dec_layers, dropout)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

class TransformerEncoderModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        return output


class TransformerDecoderModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt: Tensor, memory: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        output = self.transformer_decoder(tgt, memory, tgt_mask, src_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
