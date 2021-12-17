from datasets import *
from pandas.io.parsers import _validate_parse_dates_arg
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
from os import device_encoding, listdir
from os.path import isfile, join
import math
import sys
from lxml.html import fromstring
import lxml.html as PARSER
import string
import random
from collections import Counter
from copy import deepcopy
from pyter import ter
from torchmetrics import WER

FOLDERNAME = 'CS685-Project/s2s-decipherment-multilingual'
# sys.path.append('/content/drive/My Drive/{}/code'.format(FOLDERNAME))
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device =torch.device("cpu")

from models import *
from data_utils import get_chr_to_idx, get_idx_to_chr, get_decoded_output
from debug import debug_print, get_debug_mode, get_dev_mode

def fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs = 10):
  """
  Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
  Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
  """
  num_training_steps = epochs * len(train_dataloader)
  if get_debug_mode() and not get_dev_mode():
    epochs = 1
    num_training_steps = 1

  # lr_scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps=epochs)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
  # every=int(0.1 * len(train_dataloader))
  progress_bar = tqdm(range(num_training_steps))
  best_model = deepcopy(model.state_dict())
  best_SER = 1.0

  # Used for plotting later on
  train_loss_epochs, validation_loss_epochs, val_SER_epochs = [], [], []
  
  print("Training and validating model")
  for epoch in range(epochs):
      print("-"*25, f"Epoch {epoch + 1}","-"*25)
      
      best_SER, train_loss_list, validation_loss_list, val_SER_list = train_loop(model, optimizer, train_dataloader, loss_fn, lr_scheduler, progress_bar, val_dataloader, best_SER)
      train_loss_epochs += train_loss_list
      validation_loss_epochs += validation_loss_list
      val_SER_epochs += val_SER_list
      lr_scheduler.step()
      
      # validation_loss, SER, _ = validation_loop(model, loss_fn, val_dataloader)
      # validation_loss_list += [validation_loss]

      # if SER < best_SER:
      #   torch.save({
      #     'epoch': epoch,
      #     'model_state_dict': model.state_dict(),
      #     'optimizer_state_dict': optimizer.state_dict(),
      #     'scheduler_state_dict': lr_scheduler.state_dict(),
      #     'loss': validation_loss,
      #     'SER' : SER
      #     }, '/content/drive/My Drive/{}/output/saved_model.pth'.format(FOLDERNAME))
      
      # print(f"Training loss: {train_loss:.4f}")
      # print(f"Validation loss: {validation_loss:.4f}")
      # print(f"Validation SER: {SER:.4f}")
      # print()  
  return train_loss_epochs, validation_loss_epochs, val_SER_epochs, best_model



def train_loop(model, optimizer, train_data, loss_fn, lr_scheduler, progress_bar, val_data, best_SER):
  iter_count = 0
  total_loss = 0
  N = len(train_data)
  best_SER = best_SER
  
  # every=int(0.1 * len(train_data))
  every=5

  if get_debug_mode() and not get_dev_mode():
    every=1

  # Used for plotting later on
  train_loss_list, validation_loss_list = [], []
  val_SER_list = []

  model.train()
  for batch in train_data:
    iter_count+=1
    batch = {k: v.to(device) for k, v in batch.items()}
    ntoken = len(get_chr_to_idx())

    # debug_print(ntoken)
    # debug_print()

    X = batch['input_ids'][:,:-1].to(device)
    Y = batch['labels'][:,1:].to(device)
    X_pad_mask = model.create_pad_mask(X,27).to(device)
    Y_pad_mask = model.create_pad_mask(Y,27).to(device)
    X_att_mask = torch.zeros((X.size(1),X.size(1))).float().to(device)
    Y_att_mask = model.get_tgt_mask(X.size(1)).to(device)

    # debug_print("input_ids shape and values")
    # debug_print(batch['input_ids'].size())
    # debug_print()
    # debug_print("labels shape and values")
    # debug_print(batch['labels'].size())
    # debug_print()

    # debug_print("X shape and values")
    # debug_print(X.size())
    # debug_print()
    # debug_print("Y shape and values")
    # debug_print(Y.size())
    # debug_print()

    enc_out, dec_out = model(X,Y, X_att_mask, Y_att_mask, X_pad_mask, Y_pad_mask)

    # debug_print("Resized decoder Output shape and values")
    # debug_print(dec_out.view(-1, ntoken).size())
    # debug_print()
    
    # debug_print("Resized Y shape and values")
    # debug_print(Y.reshape(-1).size())
    # debug_print()

    lang_loss = loss_fn(enc_out, batch['lang'].view(-1))
    s2s_loss = loss_fn(dec_out.view(-1, ntoken),Y.reshape(-1))

    # debug_print("Lang Loss")
    # debug_print(lang_loss)
    # debug_print("S2S Loss")
    # debug_print(s2s_loss)
    
    if model.task == 'multi':
      loss = lang_loss/10 + s2s_loss
    else:
      loss = s2s_loss

    # debug_print("Total Loss")
    # debug_print(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    progress_bar.update(1)
    
    # total_loss += loss.detach().item()
    if iter_count%every==0 or iter_count==1 or iter_count==len(train_data):
      train_loss = loss.detach().item()
      validation_loss, SER, pred, text = validation_loop(model, loss_fn, val_data)
      best_model = 'saved_model_{}_{}_{}'.format(model.task, model.seq_len-1, model.space_enc)
      
      debug_print("Iteration" + str (iter_count))
      debug_print(train_loss)
      debug_print(validation_loss)
      debug_print(SER)

      if SER < best_SER:
        best_SER = SER
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': lr_scheduler.state_dict(),
          'SER' : SER,
          'iter_no': iter_count,
          'iters': len(train_data)  
          }, '/content/drive/My Drive/{}/output/{}.pth'.format(FOLDERNAME,best_model))
      
      print("Iteration %d/%d: | Train Loss = %0.2f | Val Loss = %0.2f | SER = %0.2f" % (iter_count,len(train_data),train_loss, validation_loss,SER))
      d = {'text': text[:5], 'pred':pred[:5]}
      df = pd.DataFrame(d)
      display(df)
      print()

      if iter_count%every==0:
        train_loss_list+=[train_loss]
        validation_loss_list += [validation_loss]
        val_SER_list += [SER]
    
  return best_SER,train_loss_list, validation_loss_list, val_SER_list

def validation_loop(model, loss_fn, dataloader):
  """
  Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
  Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
  """
  
  model.eval()
  total_symbols = 0
  errors = 0
  total_loss = 0
  pred= []
  text=[]
  
  with torch.no_grad():
    for batch in dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      ntoken = len(get_chr_to_idx())
      X = batch['input_ids'][:,:-1].to(device)
      Y = batch['labels'][:,1:].to(device)
      X_pad_mask = model.create_pad_mask(X,27).to(device)
      Y_pad_mask = model.create_pad_mask(Y,27).to(device)
      X_att_mask = torch.zeros((X.size(1),X.size(1))).float().to(device)
      Y_att_mask = model.get_tgt_mask(X.size(1)).to(device)
      
      
      total_symbols += torch.sum (X_pad_mask == False)
      
      # debug_print("Total Symbols")
      # debug_print(total_symbols)
      # debug_print()

      # debug_print("input_ids shape and values")
      # debug_print(batch['input_ids'].size())
      # debug_print(batch['input_ids'])
      # debug_print()
      # debug_print("labels shape and values")
      # debug_print(batch['labels'].size())
      # debug_print(batch['labels'])
      # debug_print()

      enc_out, dec_out = model(X,Y, X_att_mask, Y_att_mask, X_pad_mask, Y_pad_mask)
      # debug_print("dec_out shape")
      # debug_print(dec_out.size())
      # debug_print()

      loss = loss_fn(dec_out.view(-1, ntoken),Y.reshape(-1))   
      
      y_hat = torch.argmax(dec_out, dim=2, keepdim=False)
      # debug_print("y_hat shape and value")
      # debug_print(y_hat.size())
      # debug_print(y_hat)
      # debug_print()
      
      pred += get_decoded_output(y_hat.tolist())
      text+= get_decoded_output(Y.tolist())
      errors += torch.sum((y_hat!= Y)[Y_pad_mask==False]).detach().item()
      # debug_print(total_symbols)
      # debug_print(errors)
      total_loss += loss.detach().item()
      
    if model.space_enc=='with_space':
      ER = float(errors/total_symbols)
    else:
      metric = WER()
      ER = metric(' '.join(pred),' '.join(text))
  
  return (total_loss / len(dataloader)), ER, pred, text

# def predict(model, dataloader):
#   """
#   Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
#   Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#   """
#   model.eval()
  
#   y_input = torch.tensor([[28]], dtype=torch.long, device=device)

#   num_tokens = len(input_sequence[0])

#   for _ in range(max_length):
#     # Get source mask
#     tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
    
#     pred = model(input_sequence, y_input, tgt_mask)
    
#     next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
#     next_item = torch.tensor([[next_item]], device=device)

#     # Concatenate previous input with predicted best word
#     y_input = torch.cat((y_input, next_item), dim=1)

#     # Stop if model predicts end of sentence
#     if next_item.view(-1).item() == EOS_token:
#       break
  
#   model.eval()
#   total_symbols = 0
#   errors = 0
#   total_loss = 0
#   pred= []
  
#   with torch.no_grad():
#     for batch in dataloader:
#       batch = {k: v.to(device) for k, v in batch.items()}
#       ntoken = len(get_chr_to_idx())
#       X = batch['input_ids'][:,:-1].to(device)
#       Y = batch['labels'][:,1:].to(device)
#       y_input = torch.tensor([[28]], dtype=torch.long, device=device)
      
#       X_pad_mask = model.create_pad_mask(X,27).to(device)
#       Y_pad_mask = model.create_pad_mask(Y,27).to(device)
#       X_att_mask = torch.zeros((X.size(1),X.size(1))).float().to(device)
#       Y_att_mask = model.get_tgt_mask(X.size(1)).to(device)
      
      
#       total_symbols += torch.sum (X_pad_mask == False)

#       enc_out, dec_out = model(X,Y, X_att_mask, Y_att_mask, X_pad_mask, Y_pad_mask)

#       loss = loss_fn(dec_out.view(-1, ntoken),Y.reshape(-1))   
      
#       y_hat = torch.argmax(dec_out, dim=2, keepdim=False)
      
#       pred += get_decoded_output(y_hat.tolist()) 
      
      
#       errors += torch.sum((y_hat!= Y)[Y_pad_mask==False])
#       # debug_print(total_symbols)
#       # debug_print(errors)
#       total_loss += loss.detach().item()

#   return y_input.view(-1).tolist()