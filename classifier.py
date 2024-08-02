#!/usr/bin/env python
# coding: utf-8

# ### Integrando clasificadores

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from data_cleaning import clean_text, clean_df, insert_type_cols, delete_words, delete_words_df, preprocess_text, preprocess_df
from bert_embeddings import get_bert_embeddings
import joblib




def classifier(new_message, general_classifier, tokenizer , especific_classifier, df):

    new_message = clean_text(new_message)
    new_message = preprocess_text(new_message)
   
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df['type_general'])
    
    # Tokenizar el mensaje
    inputs = tokenizer(new_message, return_tensors='pt', padding=True, truncation=True)

    # Obtener la predicción
    with torch.no_grad():
        logits = general_classifier(**inputs).logits
        predicted_class = logits.argmax().item()
    
    # Decodificar la clase predicha
    
    predicted_label = label_encoder.inverse_transform([predicted_class])

    if predicted_label[0] == "other":
        focus_list = list(df.focus.unique())
        focus_list = [word for sentence in focus_list for word in sentence.split()]
    
        included_words = ["syndrome", "deformity", "body", "disease", "mouth", 
                  "nose", "ear", "congenital", "gender", "teen", "poison", "mouth",
                  "help", "to", "me", "leg", "pain", "abdominal", "a", "size", "of",
                  'bb', 'in','my','right','forearm', "havent", 'cannot','eat','losing',
                  'weight','hard','time','getting','to','sleep','not','being','able','to',
                  'breath','binge','drinker','have','trouble','walking','using','morphine',
                  'fire','burn','gives','stain','odd','alcohol','and','opiate','withdrawal',
                  'penis','did','not','grow','binocular','vision','hands','hurt','throat','heart',
                  'condition', 'prostate', "lung", "chicken", "vaginal", "night", "nerve", "disorder"] + list(df.type.unique())

        focus_list = [word for word in focus_list if word not in included_words]
        new_message = delete_words(new_message, focus_list) #El clasificador especifico se entrenó aplicando delete_words() a los mensajes
        new_embedd = get_bert_embeddings(new_message)

        pred = especific_classifier.predict([new_embedd])

        return pred[0]
    else:

        return predicted_label[0]



