from IPython.core.display import display
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

from preprocess import frequency_encode_string, get_alphabet, frequency_encode_string_with_spaces
from debug import debug_print, get_debug_mode, get_dev_mode

max_len = 258
space_enc = 'with_space'

def get_max_len():
  global max_len
  return max_len

def set_max_len(value):
  global max_len
  max_len = value

def get_space_enc():
  global space_enc
  return space_enc

def set_space_enc(value):
  global space_enc
  space_enc = value

def freq_array(line):
  max_len = get_max_len()
  space_enc = get_space_enc()

  chr_to_idx = get_chr_to_idx()
  if space_enc=='enc_space':
    freq_str = frequency_encode_string_with_spaces(line)
  
  elif space_enc=='removed_spaces':
    freq_str = frequency_encode_string(line)
    freq_str = freq_str.replace('_',' _ ')
    freq_str = freq_str.replace('_','')

  else:
    freq_str = frequency_encode_string(line)
    freq_str = freq_str.replace('_',str(chr_to_idx['_']))
  
  freq_arr = [x for x in freq_str.split()][:max_len]
  freq_arr = ['28']+freq_arr+['29']
  
  # print(max_len)
  # debug_print("Integer Array")
  # debug_print(freq_arr)
  # debug_print("")
  return ' '.join(freq_arr + [str(chr_to_idx['<PAD>'])] * (max_len + 2 - len(freq_arr)))

def alpha_enc(row):
  # debug_print("Row")
  # debug_print(row + '\n')
  max_len = get_max_len()
  # print(max_len)
  chr_to_idx = get_chr_to_idx()
  alpha_array = ['28'] + [str(chr_to_idx[x]) for x in list(row)][:max_len] + ['29']
  return ' '.join(alpha_array + [str(chr_to_idx['<PAD>'])] * (max_len +2 - len(alpha_array)))

def str_to_tensor(row):
  return torch.tensor([int(x) for x in row.split(' ')])

def get_decoded_output(output_id_arrays):
  N = len(output_id_arrays)
  idx_to_chr = get_idx_to_chr()
  decoded_sequences=[]

  for output_id_array in output_id_arrays:
    T = len(output_id_array)
    char_array = [idx_to_chr[x] for x in output_id_array]
    decoded_sequence = ' '.join(char_array).replace('<PAD>','').strip()
    decoded_sequences.append(decoded_sequence)
  
  return decoded_sequences

def create_master_data(path, task, cip_len, space_enc):
  set_max_len(cip_len)
  set_space_enc(space_enc)
  lang_labels = get_lang_labels()
  splits = {'train':[], 'test': []}
  filenames = [f for f in listdir(path) if isfile(join(path, f))]
  for filename in filenames:
    print("Loading "+filename+' ...')
    df = get_tensor_df(path+'/'+filename, task, cip_len, space_enc)
    lang = filename.split('.')
    # debug_print("Lang")
    # debug_print(type(lang))
    # debug_print(len(lang))
    # debug_print(lang)
    # debug_print('')
    df['lang'] = [lang_labels.index(lang[0])] * len(df)
    # display(df)
    if lang[1] in splits:
      splits[lang[1]].append(df)
  
  result = []
  for split,frames in splits.items():
    master_df = pd.concat(frames).sample(frac=1, random_state = 24).reset_index(drop=True)
    result.append(master_df)
    tgt_fname = '{}_{}_{}'.format(split, cip_len, space_enc)
    master_df.to_csv('/content/drive/My Drive/CS685-Project/s2s-decipherment-multilingual/master_data/'+tgt_fname+'.csv', index = False)
  
  return tuple(result)


def get_tensor_df(filename, task, cip_len, space_enc):
  nrows=1650//14
  if get_debug_mode():
    nrows = 5
  if get_dev_mode():
    nrows = 850//14
  df = pd.read_csv(filename, delimiter='\n', names=['text'], nrows=nrows)

  df['input_ids']=(df['text'].str.strip()).str.replace(' ','').map(freq_array)
  df['labels'] = (df['text'].str.strip()).str.replace(' ','').map(alpha_enc)

  # strp_lines = list(df['labels'].values)
  # debug_print("Element 0 of Stripped Lines List of Strings")
  # debug_print(df['text'].values[0] + '\n')

  # freq_lines =list(df['input_ids'].values)
  # debug_print(freq_lines[0])
  # if get_debug_mode():
  #   display(df)

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

def get_split_dfs(src_fname = '', train_split = 0.98, df=None):

  if df is not None:
    val_start = 1.0 - train_split
    df_train = df[:int(train_split * len(df))].copy()
    df_val = df[int(train_split * len(df)):].copy()
    return df_train, df_val

  df = pd.read_csv(src_fname)
  df_train = df[:int(train_split * len(df))].copy()
  df_val = df[int(train_split * len(df)):].copy()

  return df_train, df_val