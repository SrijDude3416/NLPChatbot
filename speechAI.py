#Imports
import re
import pandas as pd
import json
import multiprocessing
import string
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from time import time
from collections import defaultdict
import numpy as np
import pdb

#Import Scikit
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

#Import natural language toolkit
import nltk
nltk.download('punkt')

#Import spacy and load web model
import spacy
#Must download in terminal before loading it
spacy.load("en_core_web_sm")
from spacy.vocab import Vocab

#For timing processes
t = time()

#loading data
df = pd.read_csv("casual_data_windows.csv")
df = df.dropna().reset_index(drop=True)

#Loading spacy nlp
nlp = spacy.load("en_core_web_sm")

#Creating sentences to train w2v
sent = [row.split() for row in df['Person1']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

biggestLen = 31
currentLen = 0
index = 0
vecsX = []
vecsY = []
sent = []

#Create X and y arrays
X = []
y = []

#Check num cores in computer for w2v model input
cores = multiprocessing.cpu_count()

#Create model
w2vModel = Word2Vec(min_count=20,
                  window=2,
                  size=300,
                  sample=6e-5,
                  alpha=0.03,
                  min_alpha=0.0007,
                  negative=20,
                  workers=cores-1)

# --------------------------------------------------------------------------------------

#Build vocab in Word2Vec model
print("Starting to build vocab...")
w2vModel.build_vocab(sentences, progress_per=10000)
#Print time taken to build vocab
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# --------------------------------------------------------------------------------------

#Train Word2Vec Model
print("Starting to train Word2Vec...")
w2vModel.train(sentences, total_examples=w2vModel.corpus_count,
          epochs=20, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2vModel.init_sims(replace=True)

print(len(df['Person1']))
print("Appending Data...")

#Append Vectorized data into X and y training sets
for x in range(0, int(len(df['Person1']) / 560)):
  for i in range(0, len(df['Person1'][x].split(" "))):
      if x < len(df['Person1']) - 1 and i < len(df['Person1'][x].split(" ")):
          vecsX.append(nlp(df['Person1'][x][i].lower()).vector_norm)
          print(x)

  for e in range(0, biggestLen - len(df['Person1'][x].split(" "))):
      vecsX.append(nlp("unnapliccable").vector_norm)

  X.append(vecsX)
  vecsX = []

  for l in range(0, len(df['Person1'][x].split(" "))):
      if x < len(df['Person1']) - 1 and l < len(df['Person1'][x].split(" ")):
          vecsY.append(nlp(df['Person1'][x][l].lower()).vector_norm)
          print(x)

  for r in range(0, biggestLen - len(df['Person1'][x].split(" "))):
      vecsY.append(nlp("unnapliccable").vector_norm)

  y.append(vecsY)
  vecsY = []

print('Time to append data: {} mins'.format(round((time() - t) / 60, 2)))
# sentend = np.ones

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Create and fit model
clf = MLPRegressor().fit(X_train, y_train)

inp = input("Say something:")

for k in range(0, len(inp.split(" "))):
  sent.append(nlp(inp[k].lower()).vector_norm)

for i in range(0, biggestLen - len(inp.split(" "))):
  sent.append(nlp("unnapliccable").vector_norm)

sent = [sent]

predictions = clf.predict(sent)
pdb.set_trace()
# outputlist = [w2vModel.wv.similar_by_vector([predictions[0][i], predictions[0][i]], topn=1)[0][0] for i in range(biggestLen)]
# output = ' '.join(outputlist)
