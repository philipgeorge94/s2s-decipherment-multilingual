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

# FOLDERNAME = 'CS685-Project/s2s-decipherment-multilingual'
# sys.path.append('/content/drive/My Drive/{}/code'.format(FOLDERNAME))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from data_utils import str_to_tensor

class CipherDataset(torch.utils.data.Dataset):
  '''
  Defines the CipherDataset object type. Every CipherDataset object has a 
  __get_item__() function that returns three data elements
  1. input_ids of the frequency encoded input sequence
  2. Labels of the target sequence represented by vocabulary indices

  '''
  def __init__(self, df):
    '''
    Takes a DataFrame as input and converts the string columns to tensors
    using a map function defined in ./data_utils.py.
    The data is stored as numpy arrays which are automatically converted to
    torch.Tensor by PyTorch when __get_item__ is called by the DataLoader() 
    constructor.
    '''
    self.input_ids = np.array(df['input_ids'].map(str_to_tensor).values)
    self.labels = np.array(df['labels'].map(str_to_tensor).values)
    self.lang_labels = np.array(df['lang'].values)
        
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    '''
    This function returns the three data elements. Pytorch automatically 
    stacks them into batch-sized tensors of(batch_size, seq_len) dimensions,
    but preserves the dict() structure,
    '''
    return {'input_ids': self.input_ids[idx], 'labels':self.labels[idx], 'lang': self.lang_labels[idx]}