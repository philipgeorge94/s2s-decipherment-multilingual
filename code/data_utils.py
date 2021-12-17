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

#global variables that can be accessed in this module
max_len = 258
space_enc = 'with_space'

def get_max_len():
  '''
  Getter for max_len
  '''
  global max_len
  return max_len

def set_max_len(value):
  '''
  setter for max_len
  '''
  global max_len
  max_len = value

def get_space_enc():
  '''
  getter for space encoding scheme
  '''
  global space_enc
  return space_enc

def set_space_enc(value):
  '''
  Setter for space encoding scheme
  '''
  global space_enc
  space_enc = value

def freq_array(line):
  '''
  1. Used as a mapping function for pandas.DataFrames.Takes a string input of 
  ciphertext chars, and outputs a string of frequency encoded chars. 
  2. Behaves differently for different max_len and space_enc.
  3. Used to preprocess the input to encoder
  '''
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
  
  return ' '.join(freq_arr + [str(chr_to_idx['<PAD>'])] * (max_len + 2 - len(freq_arr)))

def alpha_enc(row):
  '''
  1. Mapping function used to convert string of plaintext chars into a string of 
  vocabulary indices
  2. Used to preprocess the target sequence input to decoder
  '''
  max_len = get_max_len()
  chr_to_idx = get_chr_to_idx()
  alpha_array = ['28'] + [str(chr_to_idx[x]) for x in list(row)][:max_len] + ['29']
  return ' '.join(alpha_array + [str(chr_to_idx['<PAD>'])] * (max_len +2 - len(alpha_array)))

def str_to_tensor(row):
  '''
  Mapping function used to convert an input string of space separated integers
  into a tensor. Used by CipherText()
  '''
  return torch.tensor([int(x) for x in row.split(' ')])

def get_decoded_output(output_id_arrays):
  '''
  1. Used by ./code/train_test.py for validation_loop()
  2. Input: (N,T) integer output list from decoder 
  3. Output: (N,) string list of deciphered plaintext
  '''
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
  '''
  Input:
  1. path: ./data/ where source files are stored
  2. task, cip_len, space_enc: current experiment settings
  
  Returns:
  1. (df_train, df_test) tuple of pandas.DataFrames

  Also writes these dataframes to ../master_data/
  '''
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
  '''
  Called by create_master_data() to process each source file
  '''
  nrows=1650//14
  if get_debug_mode():
    nrows = 5
  if get_dev_mode():
    nrows = 850//14
  df = pd.read_csv(filename, delimiter='\n', names=['text'], nrows=nrows)

  #Create input_ids and labels as string dtype columns in DataFrame
  #by calling mapping functions
  df['input_ids']=(df['text'].str.strip()).str.replace(' ','').map(freq_array)
  df['labels'] = (df['text'].str.strip()).str.replace(' ','').map(alpha_enc)

  return df.copy()

def get_chr_to_idx():
  '''
  Defines vocabulary -> index mapping
  '''
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
  '''
  Defines index -> vocabulary mapping
  '''
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
  '''
  Returns a list of all languages, to generate a language label.
  For examples 'catalan' is '0'
  '''
  return ['catalan','english','german','latin','spanish','danish','finnish','hungarian','norwegian','swedish','dutch','french','italian','portuguese']

def get_split_dfs(src_fname = '', train_split = 0.98, df=None):
  '''
  Creates train and val split DataFrames
  Input:
  1. src_fname: path to source .csv file to read DataFrame
  2. train_split: split of train and val data
  3. df: optional argument in-case train data is available in a DataFrame already

  Returns:
  df_train_df_val pandas.DataFrames
  '''

  if df is not None:
    val_start = 1.0 - train_split
    df_train = df[:int(train_split * len(df))].copy()
    df_val = df[int(train_split * len(df)):].copy()
    return df_train, df_val

  df = pd.read_csv(src_fname)
  df_train = df[:int(train_split * len(df))].copy()
  df_val = df[int(train_split * len(df)):].copy()

  return df_train, df_val