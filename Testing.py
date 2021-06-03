# -*- coding: utf-8 -*-
"""
Created on Mon May 31 05:57:53 2021

@author: Elvis Mondal
"""




import tensorflow as tf
import pandas as pd
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np

import string
import pandas as pd
import re
import nltk
import tensorflow as tf
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.layers import Embedding,LSTM,Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np


vocab_size = 10000
embedding_dim = 64
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 16000



    
txts=open('D:/DePaul University/3rd Quarter/Artificial Intelligence/FinalProject/UserReply.txt',encoding='utf-8').readlines()
lowcase=txts   

def Inputline(x):
  with open(x) as file: 
    line = []
    for lines in file.readlines():
      line.append(lines)
    return line
line = Inputline('D:/DePaul University/3rd Quarter/Artificial Intelligence/FinalProject/UserReply.txt')

words=[]

def clearword(line):
  list1,list2 = [],[]
  for lines in line:
    x,y=lines.split(";")
    y = y.replace('\n','')
    list1.append(x)
    list2.append(y)
  df = pd.DataFrame(list(list1),columns=['sentence'])
  df['emotion'] = list2
  return df 
df = clearword(line)
df.emotion.value_counts()

df.isnull().sum()


wn = WordNetLemmatizer()

def sword(x):
  cor = []
  for words in x:
    words = words.split()
    y = [wn.lemmatize(word) for word in words if not word in stopwords.words('english')]
    y =  ' '.join(y)
    cor.append(y)
  return cor
x = sword(df['sentence'])

x[:5]



xtest = sword(df['sentence'])
all = x + xtest
len(all)

y = df.iloc[:,1].values
y.shape


ytest =df.iloc[:,1].values
ytest.shape

ytrain = pd.DataFrame(y)



tokenizer = Tokenizer(nb_words=10000, split=' ')
tokenizer.fit_on_texts(all)
Xs = tokenizer.texts_to_sequences(all)
Xs = pad_sequences(Xs,maxlen=20,padding='post',truncating='post')
Ys = pd.get_dummies(ytrain).values





Xtrain = Xs[:16000]
Xtest = Xs[16000:]



Ytrain = Ys
Ytest = pd.get_dummies(ytest).values

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim = 64, input_length=20),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(6,activation='softmax')
   
])

model.summary()

model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])


model.fit(Xtrain,Ytrain,batch_size=32,epochs=20,verbose=2,validation_split=0.2)

loss,acc = model.evaluate(Xtest,Y_test)












def emt():
    sentence=[]
    Emotions=open('D:/DePaul University/3rd Quarter/Artificial Intelligence/FinalProject/UserResponse.txt',encoding='utf-8').readlines()
    for st in Emotions:
        sentence.append(st.lower())
#sentence = ["i am feeling super excited","i didnt feel humiliated"]
        sequences = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    

    a=model.predict(padded);

    b=np.round(a,7)
    #print(np.round(a,7))
    print(sentence)
    #print(b)
    for i in range(len(sentence)):
        source_data=(list(b[i]))
        #print(source_data)
    for x in source_data:
      if (x > 0.1667535):
          e="Sad"
      else:
          e="Other Emotion"    
    return e 


r=emt()      

print(r)
          
    
    
    
    
    
    