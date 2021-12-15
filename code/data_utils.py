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

def freq_array(line):
  chr_to_idx = get_chr_to_idx()
  max_len = 256
  freq_str = frequency_encode_string(line)
  freq_str = freq_str.replace('_',str(chr_to_idx['_']))
  freq_arr = [x for x in freq_str.split()]
  debug_print("Integer Array")
  debug_print(freq_arr)
  debug_print("")
  return ' '.join(freq_arr + [str(chr_to_idx['<PAD>'])] * (max_len - len(freq_arr)))

def alpha_enc(row):
  debug_print("Row")
  debug_print(row + '\n')
  max_len = 256
  chr_to_idx = get_chr_to_idx()
  alpha_array = [str(chr_to_idx[x]) for x in list(row)]
  return ' '.join(alpha_array + [str(chr_to_idx['<PAD>'])] * (max_len - len(alpha_array)))

def str_to_tensor(row):
  return torch.tensor([int(x) for x in row.split(' ')])

def get_decoded_output(output_id_arrays):
  N = len(output_id_arrays)
  idx_to_chr = get_idx_to_chr()
  decoded_sequences=[]

  for output_id_array in output_id_arrays:
    T = len(output_id_array)
    char_array = [idx_to_chr[x] for x in output_id_array]
    decoded_sequence = ' '.join(char_array).replace('<PAD>','').strip().replace(' ','')
    decoded_sequences.append(decoded_sequence)
  
  return decoded_sequences

def create_master_data(path):
  lang_labels = get_lang_labels()
  splits = {'train':[], 'test': [], 'dev':[]}
  filenames = [f for f in listdir(path) if isfile(join(path, f))]
  for filename in filenames:
    df = get_tensor_df(path+'/'+filename)
    lang = filename.split('.')
    debug_print("Lang")
    debug_print(type(lang))
    debug_print(len(lang))
    debug_print(lang)
    debug_print('')
    df['lang'] = [lang_labels.index(lang[0])] * len(df)
    splits[lang[1]].append(df)
  
  result = []
  for split,frames in splits.items():
    master_df = pd.concat(frames).sample(frac=1, random_state = 24).reset_index(drop=True)
    result.append(master_df)
    master_df.to_csv('/content/drive/My Drive/CS685-Project/s2s-decipherment-multilingual/master_data/'+split+'.csv', index = False)
  
  return tuple(result)


def get_tensor_df(filename=""):
  nrows=None
  if get_debug_mode():
    nrows = 1
  df = pd.read_csv(filename, delimiter='\n', names=['text'], nrows=nrows)
  # display(df)
  df['input_ids']=(df['text'].str.strip()).str.replace(' ','').map(freq_array)
  df['labels'] = (df['text'].str.strip()).str.replace(' ','').map(alpha_enc)

  strp_lines = list(df['labels'].values)
  debug_print("Element 0 of Stripped Lines List of Strings")
  debug_print(df['text'].values[0] + '\n')

  freq_lines =list(df['input_ids'].values)
  debug_print(freq_lines[0])
  if get_debug_mode():
    display(df)

  return df.copy()

def get_chr_to_idx():
  alphabet = get_alphabet()
  vocab = {}
  vocab['_'] = 26
  vocab['<PAD>'] = 27
  vocab['<SOS>'] = 28 
  vocab['<EOS>'] = 29
  
  for letter in alphabet:
    vocab[letter] = ord(letter) - 97
  return vocab

def get_idx_to_chr():
  alphabet = get_alphabet()
  vocab = {}
  vocab[26] = '_'
  vocab[27] = '<PAD>'
  vocab[28] = '<SOS>' 
  vocab[29] = '<EOS>'
  
  for letter in alphabet:
    vocab[ord(letter) - 97] = letter
  return vocab

def get_lang_labels():
  return ['catalan','english','german','latin','spanish','danish','finnish','hungarian','norwegian','swedish','dutch','french','italian','portuguese']

def get_split_dfs(src_fname, train_split = 0.8):
  df = pd.read_csv(src_fname)
  val_start = len(df) - 3000
  df_train = df[:val_start].copy()
  df_val = df[val_start:].copy()

  return df_train, df_val

class CipherDataset(torch.utils.data.Dataset):
  def __init__(self, df):
    self.input_ids = np.array(df['input_ids'].map(str_to_tensor).values)
    self.labels = np.array(df['labels'].map(str_to_tensor).values)
    self.lang_labels = np.array(df['lang'].values)
        
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return {'input_ids': self.input_ids[idx], 'labels':self.labels[idx], 'lang' : self.lang_labels[idx]}