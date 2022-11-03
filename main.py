#importation librairies pertinentes
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import string

#NLP
import nltk
from nltk.stem import WordNetLemmatizer
import re
import spacy
import os
import en_core_web_sm


import os

import joblib
from fastapi import FastAPI 
from pydantic import BaseModel



#Initialisation d’une instance d’application FastAPI
app = FastAPI(
    title="proposition tags",
    description="A simple API that use NLP model to predict tags",
    version="0.1",
)
#define object
class TEST(BaseModel):
    predict_tag: str


#loading counter vectorizer
count_vect = joblib.load('count_vect.pkl')

#loading multibinarizer
MultiLabelBinarizer_mlb = joblib.load('MultiLabelBinarizer_mlb.pkl')

# #loading classifier
classifier2 = joblib.load('classifier2_logistic_regression.pkl')

# #loading useless words
useless_words = joblib.load('useless_words.pkl')


#transformation sentence
def clean_text(sentence):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    # parse html content
    soup = BeautifulSoup(sentence, "html.parser")


    # return data by retrieving the tag content
    text =' '.join(soup.stripped_strings)
    
    # Make lower
    text = text.lower()
    
        # Remove line breaks
    # Note: that this line can be augmented and used over
    # to replace any characters with nothing or a space
    text = re.sub(r'\n', '', text)
    
        # Remove punctuation
    text =text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()

        # Remove stop words
    text = text.split()
    
    text_filtered = [word for word in text if not word in useless_words]
    text_filtered = [word for word in text_filtered if len(word)!=1]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    
    # Lemmatize
    lem = WordNetLemmatizer()
    text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    texts = ' '.join(text_stemmed)
    
        #clean_verb_noun
    if len(text_stemmed)==0:
        return texts
    else :
        data_words = list(elt for elt in text_stemmed)
        texts_out = []
        doc = nlp(" ".join(word for word in data_words)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
#        texts_out=' '.join(texts_out)
    return texts_out

@app.get('/')
def get_root():
    return {'message': 'Welcome to the tag prediction API'}

@app.get('/test/{sentence}',response_model=TEST)

# function for tag prédiction
async def test(sentence :str):
    """
    A simple function that receive a sentence and predict tag related to the topic.
    :param sentence:
    :return: prediction, probabilities
    caution: put sentence in doctring 
    """
     # clean the sentence
    cleaned_sentence = clean_text(sentence)

#     #BOW transform (count vectorizer)
    test_fit_count= count_vect.transform(cleaned_sentence)

#     # perform prediction
    predictions_proba2= classifier2.predict_proba(test_fit_count)
    
#     # predict tag
    df_quest_keywords_proba = pd.DataFrame(predictions_proba2, columns=list(MultiLabelBinarizer_mlb.classes_))
    proba_top_tags_test=[]
    for i, row in df_quest_keywords_proba.iterrows():
        top_tags = row.nlargest(5).index
        proba_top_tags_test.append(" ".join(top_tags))
    proba_top_tags_test=' '.join(proba_top_tags_test)        
    return TEST(predict_tag=proba_top_tags_test)