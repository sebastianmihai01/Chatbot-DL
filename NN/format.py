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

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('../intents.json').read()
intents = json.loads(data_file)


def load(intents):
    for intent in intents['intents']:
        for pattern in intent['patterns']:

            # take each word and tokenize it
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # adding documents
            documents.append((w, intent['tag']))

            # adding classes to our class list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])


"""
If you look carefully at the json file, you can see that there are sub-objects within objects. For example, 
“patterns” is an attribute within “intents”. So we will use a nested for loop to extract all of the words within 
“patterns” and add them to our words list. We then add to our documents list each pair 
of patterns within their corresponding tag. We also add the tags into our classes list,
and we use a simple conditional statement to prevent repeats.
"""


"""
nested loops -> extract words from 'patterns'
add words to -> documents (pair pattern-tag)
add tags -> classes (unique)
"""


"""
Next, we will take the words list and lemmatize and lowercase all the words inside. 
In case you don’t already know, lemmatize means to turn a word into its base meaning, or its lemma. 
For example, the words “walking”, “walked”, “walks” all have the same lemma, which is just “walk”.
The purpose of lemmatizing our words is to narrow everything down to the simplest level it can be. 
It will save us a lot of time and unnecessary error when we actually process these words for machine learning. 
This is very similar to stemming
"""


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))



"""
Now that we have our training and test data ready, we will now use a deep learning model from keras called Sequential. 
I don’t want to overwhelm you with all of the details about how deep learning models work, but if you are curious, 
check out the resources at the bottom of the article.
The Sequential model in keras is actually one of the simplest neural networks, a multi-layer perceptron.
"""