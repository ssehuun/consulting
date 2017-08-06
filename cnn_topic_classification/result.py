
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
import pytypo
train = pd.read_csv("train_topic.csv")

#question1 = train.question1

a=0
b=0
c=0
d=0
x= [0,0,0,0,0,0,0,0,0,0,0,0,0]
for num in range(len(train.is_duplicate)):
    if train.topic1[num] == train.topic2[num] and train.is_duplicate[num]==1:
#       print (train.question1[num])
#       print (train.question2[num])
#       print ("true but..")
       a = a+1
    if train.topic1[num] == train.topic2[num] and train.is_duplicate[num]==0:
#       print (train.question1[num])
#       print (train.question2[num])
#       print ("false but..")
       b += 1
    if train.topic1[num] != train.topic2[num] and train.is_duplicate[num]==1:
       print (train.question1[num])
       print (train.question2[num])
       print (train.topic1[num])
       print (train.topic2[num])

       
       print ("true but..")
       x[train.topic1[num]]+=1
       x[train.topic2[num]]+=1

       c = c+1
    if train.topic1[num] != train.topic2[num] and train.is_duplicate[num]==0:
#       print (train.question1[num])
#       print (train.question2[num])
#       print ("false but..")
       d += 1
print ("1 case :" + str(a))
print ("0 case :" + str(b))
print ("1 case :" + str(c))
print ("0 case :" + str(d))
print (x)
