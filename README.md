# s2s-decipherment-multilingual
Deciphering simple substitution ciphers with multi-task training of transformers with language loss

## Steps
1. First clone the repository on Colab. Navigate to destination folder on your machine and run the following in a Colab Notebook

```
!git clone https://github.com/philipgeorge94/s2s-decipherment-multilingual.git
```

2. If you're not running this on google colab, dependencies will have to be installed on terminal/conda/PyCharm venv etc
These include, but are not limited to, the following four.
```
$ pip install transformers --quiet
$ pip install datasets transformers[SentencePiece] --quiet
$ pip install pyter3 --quiet
$ pip install torchmetrics --quiet
```

3. Open **baseline_language_loss.ipynb** and run the 4 cells in the **'Dependencies'** section
4. Other instructions are provided in the Notebook comments in each cell, but in brief:
5. Experiment settings - cipher length, model/task type, space encoding scheme - have to be set in initial cell
6. Train,Val, and Test data have to be either generated afresh or loaded from ./master_data/ depending on whether the parrticular experiment has been run before
7. The remaining cells can be run one after another

##Folders and Files
> **Note:** Other files may be present in the folders, but those are not strictly necessary for the project
### / 'root'
1. **baseline_language_loss.ipynb** contains the main code for experiments and should be the entry point for the user
2. **preprocessing.ipynb** contains code we used for preprocessing the raw corpora
3. **DataAnalysis.ipynb** contains some code we used to analyse the data

### code/ 
1. Contains all the .py files used in the main notebook
2. **data.py** contains the CipherDataset() object
3. **data_utils.py** contains util functions for loading and preparing input data
4. **models.py** defines the Deciphormer() transformer model
5. **preprocess.py** contains code used for preprocessing the corpora
6. **train_test.py** contains the main train and validation functions

###master_data/
1. Contains the 12 cached files = 2 train_test_splits * 2 cipher_lengths * 3 space_encoding_schemes