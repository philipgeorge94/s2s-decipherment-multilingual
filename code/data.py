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
  def __init__(self, df):
    self.input_ids = np.array(df['input_ids'].map(str_to_tensor).values)
    self.labels = np.array(df['labels'].map(str_to_tensor).values)
    self.lang_labels = np.array(df['lang'].values)
        
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return {'input_ids': self.input_ids[idx], 'labels':self.labels[idx], 'lang': self.lang_labels[idx]}