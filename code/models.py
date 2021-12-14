from datasets import *
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, \
    get_scheduler
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_utils import *
from solver import *
from PositionalEncoding import *
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math
import sys

FOLDERNAME = 'CS685-Project/s2s-decipherment-multilingual'
sys.path.append('/content/{}'.format(FOLDERNAME))

dropout_prob = 0

class Deciphormer(torch.nn.Module):
    '''
    Defining the base model:
    1)
    '''

    def __init__(self, ntoken: int = 30, d_model: int = 512, nhead: int = 8, d_hid: int = 2048, nlayers: int = 6,
                 dropout: float = 0.5):
        # Initialize model attributes
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout

        # Define model layers

        self.embedder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(self.d_model, nhead, d_hid, batch_first = True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

    def forward(self, data: Tensor, mask: Tensor) -> Tensor:
      src = self.embedder(src) * math.sqrt(self.d_model)
      src = self.pos_encoder(src)
      out1 = self.transformer_encoder(src, mask)
      out2 = self.decoder(out1)
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



