# -*- coding: utf-8 -*-
"""
Created on Sun May 16 00:26:17 2021

@author: Elvis Mondal
"""

import string
from collections import Counter
txts=open('D:/DePaul University/3rd Quarter/Artificial Intelligence/FinalProject/sample.txt',encoding='utf-8').read()
lowcase=txts.lower()

ftxt=lowcase.translate(str.maketrans('','',string.punctuation))

splitwords=ftxt.split()
print(splitwords)

stopword = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

words=[]

for word in splitwords:
    if word not in stopword:
        words.append(word)
        
print(words)

emlist=[]
with open('D:/DePaul University/3rd Quarter/Artificial Intelligence/FinalProject/emotion.txt','r') as file:
    for row in file:
        cleanline=row.replace("\n",'').replace(",",'').replace("'",'').strip()
        letter,emotions=cleanline.split(':')
        
        
        if letter in words:
            emlist.append(emotions)

print(emlist)
c=Counter(emlist)
print(c)
print(c.most_common(1)[0][0] if c else None)
        
        