
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Pdf")

#import matplotlib.pyplot as plt
import os
import time, sys
import datetime
import data_helpers
import pickle

DATA_PATH = "../real_data/"
CLIENT_DATA_PATH = DATA_PATH + 'new_cli/'

client_list = os.listdir(CLIENT_DATA_PATH)
whole_document = []
for page_title in client_list :
    with open(CLIENT_DATA_PATH + page_title, 'r', encoding="utf-8") as cli:
        docs= cli.read()
        whole_document.append(docs)
        cli.close()
print (len(whole_document))
#print(whole_document) 
final_topic = pd.read_csv('final_result.csv')
print (len(final_topic))
#batches = list (zip(whole_document,final_topic))
batches = list (zip(final_topic.values, whole_document))

topic_pickle = open(DATA_PATH + 'Topic_List', 'rb')
pickling = pickle.load(topic_pickle)
print (pickling)
temp = 0
for temp in range(5):
    flag = False
    index = 0
    checker = batches[0][0][0]
    for i in range(2500):
        checker = batches[i][0][0]
        if (temp== checker):
#            print (temp)
            
            flag = True
            index = i
        
    if (flag == True):
        print (temp)
        print (pickling[index])
        print (batches[index][1])
        print ()
    flag = False
    index = 0
    for i in range(1500):
        checker = batches[i][0][0]
        if (temp== checker):
#            print (temp)
            
            flag = True
            index = i
        
    if (flag == True):
        print (temp)
        print (pickling[index])
        print (batches[index][1])
        print ()

   

#print (batches[:10])
