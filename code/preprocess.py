from lxml.html import fromstring
import lxml.html as PARSER
import string
import random
from collections import Counter


#mainly for gigaword to parse from Gigaword SGML files to data like Gutenberg

def preprocess_to_file(read_filename, write_filename):
    lines = get_text_as_list_from_gigaword(read_filename)
    preprocessed_lines = preprocess_lines(lines)
    write_to_file(preprocessed_lines, write_filename)
    
def get_text_as_list_from_gigaword(filename):
    f = open(filename)
    f = f.read()
    root = PARSER.fromstring(f)
    para_list=[]
    for ele in root.getiterator():
        if ele.tag == "p":
          para_list.append(str(ele.text_content()))    
    return para_list
    
def preprocess_lines(para_list):    
    #lower-case all strings
    para_list = [each_string.lower() for each_string in para_list]

    #remove punctuation, newlines and digits. Replace spaces with underscores
    for i in range(len(para_list)):
      para_list[i] = "".join([char for char in para_list[i] if char not in string.punctuation])
      para_list[i] = para_list[i].replace("\n"," ")
      para_list[i] = ''.join([i for i in para_list[i] if not i.isdigit()])
      para_list[i] = para_list[i].strip()
      para_list[i] = para_list[i].replace(" ","_")
      para_list[i] = para_list[i].replace("__","_")
      if len(para_list[i]) > 256:
        s = para_list[i][:256]
        index = s.rfind("_")
        para_list[i] = s[:index] #string should not end in between a word
    return para_list
    
def write_to_file(para_list, write_filename):
    with open(write_filename, "w+") as fileptr:
      for s in para_list:
            temp_s = " ".join(s)
            fileptr.write("%s\t" % temp_s)
            fileptr.write("\n\n")

#Substitution Cipher helpers

alphabet = 'abcdefghijklmnopqrstuvwxyz'
#plaintext = "Hey_this_is_really_fun"

def get_alphabet():
    return alphabet
    
#Generate Random Key
def makeKey(alphabet):
   alphabet = list(alphabet)
   random.shuffle(alphabet)
   #return ''.join(alphabet)
   keyMap = dict(zip(alphabet, key))
   keyMap['_'] = '_'
   return keyMap
   
#Encrypt with key
def encrypt(plaintext, keyMap):
    #keyMap = dict(zip(alphabet, key))    
    return ''.join(keyMap.get(c.lower(), c) for c in plaintext)

#Decrypt with key
def decrypt(cipher, keyMap):
    #keyMap = dict(zip(key, alphabet))
    return ''.join(keyMap.get(c.lower(), c) for c in cipher)

#Frequency encoding returned as string
def frequency_encode_string(input_string):
    c_string = input_string.replace(" ","")
    c_string = "".join(input_string.split('_'))
    x = Counter(c_string)
    l = x.most_common()
    freq = {}
    for i in range(len(l)):
      freq[l[i][0]] = i
    freq_enc = ''
    for c in input_string:
      freq_enc += '_' if c == '_' else str(freq[c])
      freq_enc += " "
    return freq_enc #this is space separated for readability

#Frequency Encoding returned as a list of chars
def frequency_encode_char_list(input_string):
    c_string = input_string.replace(" ","")
    c_string = "".join(cipher.split('_')) 
    x = Counter(c_string)
    l = x.most_common()
    freq = {}
    for i in range(len(l)):
      freq[l[i][0]] = i
    freq_enc = []
    for c in cipher:
      freq_enc.append('_' if c == '_' else str(freq[c]))
    return freq_enc

#Helper function to convert strings to list of chars   
def string_as_list(input_string):
    c_string = input_string.replace(" ","")
    return list(c_string)