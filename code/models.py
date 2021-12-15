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
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, Embedding, TransformerDecoderLayer, TransformerDecoder
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

FOLDERNAME = 'CS685-Project/s2s-decipherment-multilingual'
sys.path.append('/content/drive/My Drive/{}/code'.format(FOLDERNAME))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from PositionalEncoding import *
from preprocess import *
from models import *
from data_utils import *
from data import *
from debug import *
from train_test import *

class Deciphormer(torch.nn.Module):
    '''
    Defining the base model:
    1)
    '''

    def __init__(self, ntoken: int = 30, d_model: int = 512, nhead: int = 8, d_hid: int = 2048, nlayers: int = 6,
                 dropout: float = 0.0):
        # Initialize model attributes
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.best_val_acc = -1

        # Define model layers

        self.embedder = Embedding(ntoken, d_model, padding_idx = 27)
        self.pos_encoder = PositionalEncoding(d_model, max_len = d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.embedder2 = Embedding(ntoken, d_model, padding_idx = 27)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first = True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.linearout = torch.nn.Linear(d_model, ntoken)

    def forward(self, input_ids, labels, mask=None):
      src = self.embedder(input_ids) * math.sqrt(self.d_model)
      # print(src.shape)
      src = self.pos_encoder(src)
      # print(src.shape)
      out1 = self.transformer_encoder(src, mask)
      # print(out1.shape)

      embed_tgt = self.embedder2(labels) * math.sqrt(self.d_model)
      out2 = self.transformer_decoder(embed_tgt, out1)
      out2 = self.linearout(out2)
      return (out1, out2)

class MLC(Deciphormer):
  '''
  Multitasking with Language Classification
  '''

  def __init__(self, fc_dim: int = 256, langenc_type='bert-base-multilingual-cased', langenc_pos ='after', nouts: int = 14, **kwargs):
        # Initialize model attributes
        super(MLC, self).__init__(**kwargs)
        self.fc_dim = fc_dim
        self.langenc_type = langenc_type
        self.langenc_pos = langenc_pos

        # Define additional layers for MLC setup
        if self.langenc_type != 'custom':
            self.language_encoder = AutoModel.from_pretrained(self.langenc_type)
            self.lenc_tokenizer = AutoTokenizer.from_pretrained(self.langenc_type)
            self.lenc_out_size = 768
            self.get_lenc_out = lambda x: x['last_hidden_state'][:,0,:]
            
        else:
            langenc_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
            self.language_encoder = TransformerEncoder(langenc_layers, nlayers)
            self.lenc_tokenizer = lambda x: x
            self.lenc_out_size = self.d_model
            self.get_lenc_out = lambda x: x
        self.language_classifier = nn.Sequential(
            nn.Linear(self.lenc_out_size, fc_dim),
            nn.ReLu(),
            nn.Dropout(p = self.dropout),
            nn.Linear(fc_dim, nouts)
            )
        

  def forward(self, data: Tensor, lenc_mask, **kwargs) -> Tensor:
    out_inter, final_out1 = super(MCL,self).forward(data, **kwargs)
    final_out2 = self.lang
    
    return (out1, out2)



