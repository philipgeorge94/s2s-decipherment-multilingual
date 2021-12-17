from datasets import *
from transformers import AutoTokenizer, AutoModel, AdamW, DataCollatorWithPadding, get_scheduler, logging
import pandas as pd
import torch
from torch import Tensor
from torch.optim import *
import torch.optim
from torch.utils.data import DataLoader
from torch.nn import *
from torch.nn.functional import one_hot as get_one_hot_enc
from torch.nn.functional import cross_entropy
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, Embedding, TransformerDecoderLayer, TransformerDecoder, Transformer
from tqdm.auto import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
import math
import sys
from lxml.html import fromstring
import lxml.html as PARSER
import string
import random
from collections import Counter

# FOLDERNAME = 'CS685-Project/s2s-decipherment-multilingual'
# sys.path.append('/content/drive/My Drive/{}/code'.format(FOLDERNAME))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from PositionalEncoding import PositionalEncoding

class Deciphormer(torch.nn.Module):
    '''
    Defining the base model with the following components
    1) Encoder Embedder: With padding-index=27. Output: (N,S,E)
    2) Decoder Embedder: With padding-index=27
    3) Positional Encoding: defined in ./PositionalEncoding.py
    4) Transformer Encoder
    5) Transformer Decoder
    6) language_classifier: FC Network to classify encoder output into 14 languages
    6) linear_out: To convert (N,T,E) decoder output to (N,T,ntoken)
    '''

    def __init__(self, ntoken: int = 30, d_model: int = 512, nhead: int = 8, d_hid: int = 2048, nlayers: int = 6,
                 dropout: float = 0.5, task = 'single', max_len = 256, space_enc='with_space'):
        # Initialize model attributes
        super().__init__()
        self.d_model = d_model # model size
        self.nhead = nhead #no. of attention heads
        self.d_hid = d_hid #hidden size of FF layer
        self.nlayers = nlayers #no. of encoder/decoder layers
        self.dropout = dropout #dropout
        self.best_val_acc = -1 
        self.task = task #task in ['single', 'multi']
        self.seq_len = max_len + 1 #cipher length + 1 (to account for SOS/EOS tokens)
        self.space_enc = space_enc #space encoding scheme of experiment

        # Define model layers
        self.embedder = Embedding(ntoken, d_model, padding_idx = 27)
        self.pos_encoder = PositionalEncoding(d_model, max_len = d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout=dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.language_classifier = torch.nn.Linear(self.d_model * self.seq_len, 14)

        self.embedder2 = Embedding(ntoken, d_model, padding_idx = 27)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout=dropout, batch_first = True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.linearout = torch.nn.Linear(d_model, ntoken)

        self.init_weights()
    
    def init_weights(self) -> None:
      '''
      Intialize weights of embedders and linear layers
      '''
        initrange = 0.01
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.embedder2.weight.data.uniform_(-initrange, initrange)
        
        self.language_classifier.weight.data.uniform_(-initrange, initrange)
        self.language_classifier.bias.data.zero_()
        self.linearout.weight.data.uniform_(-initrange, initrange)
        self.linearout.bias.data.zero_()

    def forward(self, input_ids, labels, X_att_mask = None, Y_att_mask = None, X_pad_mask = None, Y_pad_mask = None):
      '''
      Input:
      1. Input_ids (N,S) and labels (N,T)
      2. X_att_mask: (S,S) mask of zeros for full encoder self-attention
      3.  Y_att_mask: (T,T) upper triangular mask for causal decoder attention
      4. X_pad_mask, Y_pad_mask: (N,S) and (N,T) masks to indicate pad token positions
      '''
      src = self.embedder(input_ids) * math.sqrt(self.d_model)
      
      src = self.pos_encoder(src)
      
      out1 = self.transformer_encoder(src, mask=X_att_mask, src_key_padding_mask = X_pad_mask)
      N = out1.size(0)
      

      embed_tgt = self.embedder2(labels) * math.sqrt(self.d_model)
      out2 = self.transformer_decoder(embed_tgt, out1, tgt_mask = Y_att_mask, tgt_key_padding_mask=Y_pad_mask)
      out2 = self.linearout(out2)
      out1 = self.language_classifier(out1.view(N, -1))
      return (out1, out2)
    
    def get_tgt_mask(self, size) -> torch.tensor:
      '''
      Function to return decoder attention masks
      '''
      # Generates a squeare matrix where the each row allows one word more to be seen
      mask = torch.tril(torch.ones((size, size)) == 1) # Lower triangular matrix
      mask = mask.float()
      mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
      mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
      
      # EX for size=5:
      # [[0., -inf, -inf, -inf, -inf],
      #  [0.,   0., -inf, -inf, -inf],
      #  [0.,   0.,   0., -inf, -inf],
      #  [0.,   0.,   0.,   0., -inf],
      #  [0.,   0.,   0.,   0.,   0.]]
      
      return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
      '''
      Generate pad masks for an (N,S) tensor given a padding token ID
      '''
      # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
      # [False, False, False, True, True, True]
      return (matrix == pad_token)