#!/usr/bin/env python
# coding: utf-8

# ### Generate Embeddings

# In[14]:


import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from data_cleaning import clean_text, clean_df, preprocess_text, preprocess_df



# Cargar el modelo y el tokenizador de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedd_model = BertModel.from_pretrained('bert-base-uncased')

# Función para obtener los embeddings de BERT
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = embedd_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()




