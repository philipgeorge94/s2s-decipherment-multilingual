from datasets import *
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, \
    get_scheduler
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import *
from solver import *
from PositionalEncoding import *
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

dropout_prob = 0


class base_model(torch.nn.Module):
    '''
    Defining the base model:
    1)
    '''

    def __init__(self, ntoken: int = 30, d_model: int = 512, nhead: int = 8, d_hid: int = 2048, nlayers: int = 6,
                 dropout: float = 0.5, fc_dim: int = 256, langenc_type='bert-base-multilingual-cased', langenc_pos ='after',
                 enc_stage='after'):
        # Initialize model attributes
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.fc_dim = fc_dim
        self.langenc_type = langenc_type
        self.langenc_pos = langenc_pos
        self.fc_dim = fc_dim

        # Define model layers

        self.pos_encoder = PositionalEncoding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(self.d_model, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        if self.langenc_type != 'custom':
            self.language_encoder = AutoModel.from_pretrained(self.langenc_type)
        else:
            langenc_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
            self.language_encoder = TransformerEncoder(langenc_layers, nlayers)