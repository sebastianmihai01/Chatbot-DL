"""
We have a whole bunch of libraries like nltk (Natural Language Toolkit), which contains a whole bunch of tools for
cleaning up text and preparing it for deep learning algorithms, json, which loads json files directly into Python,
pickle, which loads pickle files, numpy, which can perform linear algebra operations very efficiently, and keras,
which is the deep learning framework we’ll be using.
"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

"""
Using Keras sequential neural network -> create a model
"""


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('../intents.json').read()
intents = json.loads(data_file)

