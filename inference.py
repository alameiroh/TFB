#!/usr/bin/env python
# coding: utf-8

# ### Medical question router

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
#from sklearn.preprocessing import LabelEncoder
from data_cleaning import clean_text, preprocess_text
#from bert_embeddings_1 import get_bert_embeddings
#from similarity_search_2 import find_similar_message
from classifier import classifier
import joblib

# Cargamos los datos:

df_complete = pd.read_csv("data/data_complete.csv", header=0)
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
df_complete["embedd_message"] = df_complete.message.apply(similarity_model.encode)


# Cargar el modelo entrenado para clasificación especifica:
especific_classifier = joblib.load('random_forest_especific_classifier.pkl')

# Cargar el modelo preentrenado para clasificación general:
general_classifier_path = './model_weights'  
general_classifier = DistilBertForSequenceClassification.from_pretrained(general_classifier_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

### Función para encontrar mensajes similares a la consulta del usuario:

def find_similar_message(new_message, df, sim_model_name  = 'sentence-transformers/all-MiniLM-L6-v2'):
    
    df = df.drop_duplicates(subset = "message", keep = "first")

    similarity_model = SentenceTransformer(sim_model_name)
    
    embedded_messages = df.reset_index().embedd_message
    
    new_message = clean_text(new_message)
    new_message = preprocess_text(new_message)
    
    encoded_new_message = similarity_model.encode(new_message)
    
    # Calcula la similitud del coseno
    similarities = util.pytorch_cos_sim(encoded_new_message, embedded_messages)
    
    # Encuentra los índices de los 5 mensajes más similares
    most_similar_indices = similarities[0].argsort(descending=True)[:5]
    most_similar_Qids = []

    for i in list(most_similar_indices):
        most_similar_Qids.append(df.iloc[int(i)].Q_id)

    return most_similar_Qids


### Función del enrutador 


def router(new_message, general_classifier, tokenizer , especific_classifier, df):

    new_message_type = classifier(new_message, general_classifier, tokenizer , especific_classifier, df)

    similar_qs = find_similar_message(new_message, df)
    similarity_df = pd.concat([df[df['Q_id'] == i] for i in similar_qs])

    Q_A = similarity_df[similarity_df["type"] == new_message_type]

    #Asignar mensajes  de otros tipos en caso de que no haya del mismo tipo entre los más similares:
    if Q_A.shape[0] == 0:

        Q_A = similarity_df.iloc[:5]
        
    router_response = ""
    for i in range(Q_A.shape[0]):
        
        #router_response += f"Possible Q&A nº{i+1}:\nFocus: {similarity_df.focus.iloc[i]}\nQuestion: {similarity_df.message.iloc[i]}\n\nAnswer: {similarity_df.answer.iloc[i]}\n\n"
        router_response += f"Possible Q&A nº{i+1}:\n\nAnswer: {similarity_df.answer.iloc[i]}\n\n"
    
    return router_response
    

 ### Interacción con el usuario:   

cont = "Yes"
print("Hi, I'm a chatbot for medical questions.\n")
while cont == "Yes":
    print("Which is your question? ", end="")
    query = input()
    print(f"\n\nThis are some possible answers to your question, in our database:\n\n{router(query, general_classifier, tokenizer , especific_classifier, df_complete)}")
    print("Any more questions? Introduce; Yes o No: ", end="")
    cont = input()



