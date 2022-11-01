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





