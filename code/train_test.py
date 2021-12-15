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

def fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    num_training_steps = epochs * len(train_data)
    lr_scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    every=int(0.1 * len(train_data))
    progress_bar = tqdm(range(num_training_steps))

    if get_debug_mode():
      epochs = 1
      num_training_steps = 1
      every = 1

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, optimizer, loss_fn, train_dataloader, progress_bar)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list



def train_loop(model, optimizer, train_data, loss_fn, lr_scheduler, progress_bar):
  iter_count = 0
  print()
  print("Started Epoch %d/%d:" % (epoch+1,epochs))
  for batch in train_data:
    model.train()
    iter_count+=1
    batch = {k: v.to(device) for k, v in batch.items()}
    
    output = model(**batch)
    # print(output[0].size())
    # print(output[1].size())
    ntoken = len(get_chr_to_idx())
    # print(batch['labels'].size())
    loss = cross_entropy(output[1].view(-1, ntoken),batch['labels'].view(-1))
    
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
    
    if iter_count%every==0 or iter_count==1 or iter_count==len(train_data):
      SER= eval_model(model, val_data)
      print("Iteration %d/%d: | Loss = %0.2f | %s" % (iter_count,len(train_data),loss.item(), SER))
    
    if get_debug_mode():
      break
  return best_model

def eval_model(model, data):
  model.eval()
  total_symbols = 0
  errors = 0
  pred = []
  metric = load_metric('bleu')
  for batch in data:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      ntoken = len(get_chr_to_idx())
      # print(batch['labels'].size())
      total_symbols += int(batch['labels'].view(-1).size()[0])
      output = model(**batch)
      # print(output[1].size())
      y_hat = torch.argmax(output[1], dim=2, keepdim=False)
      # print(y_hat.size())

      # print(y_hat.view(-1).size())
      # print(batch['labels'].view(-1).size())
      
      errors += torch.sum(y_hat.view(-1)!= batch['labels'].view(-1))
      # decoded_seqs = get_decoded_output(y_hat.tolist())
      # labels = get_decoded_output(batch['labels'].tolist())
      # pred+= decoded_seqs

      # metric.add_batch(predictions=decoded_seqs, references=labels)
      # print(labels)
  SER = float(errors/total_symbols)
  return SER