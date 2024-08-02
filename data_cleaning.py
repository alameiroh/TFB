#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    
    contractions = [
    r"won\'t", r"I\'m", r"can\'t", r"don\'t", r"doesn\'t", r"shouldn\'t", r"needn\'t", r"hasn\'t", r"haven\'t",
    r"weren\'t", r"mightn\'t", r"didn\'t", r"aren\'t", r"isn\'t", r"wouldn\'t", r"couldn\'t",
    r"wasn\'t", r"it\'s", r"you\'re", r"he\'s", r"she\'s", r"we\'re", r"they\'re", r"I\'d",
    r"you\'d", r"he\'d", r"she\'d", r"we\'d", r"they\'d", r"I\'ll", r"you\'ll", r"he\'ll",
    r"she\'ll", r"we\'ll", r"they\'ll", r"I\'ve", r"you\'ve", r"we\'ve", r"they\'ve", r"\'m"]

    full_words = [
    " will not"," I am", " cannot", " do not", " does not", " should not", " need not", " has not",
    " have not", " were not", " might not", " did not", " are not", " is not",
    " would not", " could not", " was not", " it is", " you are", " he is", " she is",
    " we are", " they are", " I would", " you would", " he would", " she would",
    " we would", " they would", " I will", " you will", " he will", " she will",
    " we will", " they will", " I have", " you have", " we have", " they have", " am"]
    
    for contraction, full_word in zip(contractions, full_words): #Change contractions with full words
        text = re.sub(contraction, full_word, text)
    
    text = re.sub(r"\'s","", text) # Delete apostrophes
    text = re.sub(r"\r\n"," ", text) 
    
    text = re.sub(r"[^A-Za-z\s]", "", text) # Delete special characters
    
    text = text.lower() # Change to lowercase letters
    text = re.sub(r"[ ]+" , " " , text) # Delete multiple spaces
    
    return text
    
    
def clean_df(df):
    df_clean = df.copy()
    for col in ["message", "type", "focus"]:
        df_clean[col] = df_clean[col].apply(clean_text)
    return df_clean


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    
    
    # Tokenizar el texto
    tokens = nltk.word_tokenize(text)
    
    # Eliminar stopwords y lematizar
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Unir tokens en un solo string
    processed_text = ' '.join(cleaned_tokens)
    
    return processed_text



def preprocess_df(df):
    
    df_preprocessed = df.copy()

    df_preprocessed.message = df_preprocessed.message.apply(preprocess_text)

    return df_preprocessed

    

def insert_type_cols(df):

    type_general_map = {'association': 'other',
     'cause': 'other',
     'complication': 'other',
     'contraindication': 'other',
     'diagnoseme': 'other',
     'diagnosis': 'other',
     'dosage': 'other',
     'genetic changes': 'other',
     'indication': 'other',
     'information': 'information',
     'ingredient': 'other',
     'inheritance': 'other',
     'interaction': 'other',
     'organization': 'other',
     'prevention': 'other',
     'prognosis': 'other',
     'side effects': 'other',
     'storage and disposal': 'other',
     'susceptibility': 'other',
     'symptom': 'other',
     'tapering': 'other',
     'treatment': 'treatment',
     'usage': 'other',
     'association': 'other',
     'resources': 'other'}

    type_especific_map = {'association': 'association',
     'cause': 'cause',
     'complication': 'other',
     'contraindication': 'indication',
     'diagnoseme': 'diagnosis',
     'diagnosis': 'diagnosis',
     'dosage': 'indication',
     'genetic changes': 'other',
     'indication': 'indication',
     'information': 'information',
     'ingredient': 'other',
     'inheritance': 'other',
     'interaction': 'indication',
     'organization': 'organization',
     'prevention': 'other',
     'prognosis': 'prognosis',
     'side effects': 'other',
     'storage and disposal': 'indication',
     'susceptibility': 'susceptibility',
     'symptom': 'symptom',
     'tapering': 'other',
     'treatment': 'treatment',
     'usage': 'indication',
     'resources': 'organization'}
                         
    
    df_complete = df.copy()
    df_complete
    type_especific = df_complete.type.map(type_especific_map)
    type_general = df_complete.type.map(type_general_map)

    df_complete.insert(df_complete.columns.get_loc("type") + 1, "type_especific", type_especific)
    df_complete.insert(df_complete.columns.get_loc("type") + 2, "type_general", type_general)

    return df_complete



def delete_words(text, words):
    
    return " ".join([word for word in text.split() if word not in words])


def delete_words_df(df):
    
    focus_list = list(df_complete.focus.unique())
    focus_list = [word for sentence in focus_list for word in sentence.split()]
    
    included_words = ["syndrome", "deformity", "body", "disease", "mouth", 
                 "nose", "ear", "congenital", "gender", "teen", "poison", "mouth",
                  "help", "to", "me", "leg", "pain", "abdominal", "a", "size", "of",
                  'bb', 'in','my','right','forearm', "havent", 'cannot','eat','losing',
                  'weight','hard','time','getting','to','sleep','not','being','able','to',
                  'breath','binge','drinker','have','trouble','walking','using','morphine',
                  'fire','burn','gives','stain','odd','alcohol','and','opiate','withdrawal',
                  'penis','did','not','grow','binocular','vision','hands','hurt','throat','heart',
                  'condition', 'prostate', "lung", "chicken", "vaginal", "night", "nerve", "disorder"
                 ] + list(df_complete.type.unique())

    focus_list = [word for word in focus_list if word not in included_words]
    
    df_del_focus = df.copy()
    df_del_focus.message = df_del_focus.message.apply(lambda x: delete_words(x, focus_list))

    return df_del_focus



