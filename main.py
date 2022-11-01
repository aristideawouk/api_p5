#importation librairies pertinentes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import gensim

#NLP
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize,sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import spacy
import os
#!{sys.executable} -m spacy download en_core_web_sm
import en_core_web_lg, en_core_web_sm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

#check algo LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import os
# !{sys.executable} -m pip install transformers
import transformers
from transformers import *


from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 



#Initialisation d’une instance d’application FastAPI
app = FastAPI(
    title="proposition tags",
    description="A simple API that use NLP model to predict tags",
    version="0.1",
)

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
    if len(texts)==0:
        return texts
    else :
        data_words = list(elt for elt in text_stemmed)
        texts_out = []
        doc = nlp(" ".join(word for word in data_words)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    
    return texts_out


#loading counter vectorizer

count_vect = joblib.load(count_vect.pkl)


#loading multibinarizer

MultiLabelBinarizer_mlb = joblib.load(MultiLabelBinarizer_mlb.pkl)

#loading multibinarizer

classifier2 = joblib.load(classifier2.pkl)
