import os
import subprocess
import gevent
import multiprocessing
import data_helpers
import pickle
from collections import Counter
import operator

print("starting")
DATA_ROOT_PATH = '/share/'
PICKLE_DATA_PATH = './pickle/'

cpu_1st = 16
cpu_2nd = 16

cwd = os.getcwd()
def command1 (i):
    result = subprocess.check_output('python tokenizing.py ' + str(cpu_1st) + ' ' + str(i), shell=True)
    print("here")
def command2 (i):
    result = subprocess.check_output('python token2.py nominalized_document_dict' + str(i), shell=True)
    print ("here")
def command3():
    result = subprocess.check_output('python token3.py')
    result = subprocess.check_output('python tfidf.py')

    
print("1st job")
pool = multiprocessing.Pool(processes=cpu_1st)

pool.map(command1, range(0,cpu_1st))

print("join word dic")

word_frequency_dict = {}
word_frequency = ''

for i in range(cpu_1st):
    with open(PICKLE_DATA_PATH+'word_dic%d'%(i), 'rb') as handle:
        word_frequency_dict[i] = pickle.load(handle)

for i in range(cpu_1st):
    if (i==0):
        word_frequency = Counter(word_frequency_dict[i])
        continue
    word_frequency += Counter(word_frequency_dict[i])

word_frequency = dict(word_frequency)

# print (sorted(word_frequency.items(), key = operator.itemgetter(1), reverse = True)[:30])

with open(PICKLE_DATA_PATH + 'word_dic', 'wb') as handle:
    pickle.dump(word_frequency, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    
print ("Find Useless Words")
useless_wordlist = []
for word in word_frequency:
    if( int(word_frequency[word]) <2:
       useless_wordlist.append(word)
    if (len(word) <=1):
       useless_wordlist.append(word)
useless_wordlist = list(set(useless_wordlist))
stop_lists = data_helpers.load_stoplist()
useless_wordlist += stop_lists['general']
#print(useless_wordlist)
       
with open(PICKLE_DATA_PATH+'delete_list', 'wb') as handle:
    pickle.dump(useless_wordlist, handle, protocol = pickle.HIGHEST_PROTOCOL)

raw_document_dict = {}

for i in range(cpu_1st):
    if (i == 0):
       with open(PICKLE_DATA_PATH+'raw_document_dict%d'%(i), 'rb') as handle:
           raw_document_dict = pickle.load(handle)
       continue
    with open(PICKLE_DATA_PATH+'raw_document_dict%d'%(i), 'rb') as handle:
       raw_document_dict.update(pickle.load(handle))

nominalized_document_dict = {}

for i in range(cpu_1st):
    if( i ==0):
       with open(PICKLE_DATA_PATH+'nominalized_document_dict%d'%(i), 'rb') as handle:
           nominalized_document_dict = pickle.load(handle)
       continue
    with open(PICKLE_DATA_PATH+'nominalized_document_dict%d'%(i), 'rb) as handle:
           nominalized_document_dict.update(pickle.load(handle))
with open(PICKLE_DATA_PATH + 'reduced_whole_document', 'wb') as handle:
    pickle.dump(nominalized_document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("2nd")
pool = multiprocessing.Pool(processes=cpu_2nd)
pool.map(command2, range(0,cpu_2nd))

              
              
              
reduced_document_dict = {}
for i in range(cpu_1st):
    if (i == 0):
         with open(PICKLE_DATA_PATH+'nominalized_document_dict%d_revised'%(i), 'rb') as handle:
              reduced_document_dict = pickle.load(hanlde)
         continue
    with open(PICKLE_DATA_PATH+'nominalized_document_dict%d_revised'%(i), 'rb') as handle:
         reduced_document_dict.update(pickle.load(hanlde))
             
with open(PICKLE_DATA_PATH+'reduced_document_dict', 'wb') as handle:
    pickle.dump(reduced_document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
for word in useless_wordlist:
   try :
       del word_frequency[word]
   except:
       continue
with open(PICKLE_DATA_PATH+'word_dic', 'wb') as handle:
    pickle.dump(word_frequency, handle, protocol = pickle.HIGHEST_PROTOCOL)

word_index = 0
word_indexing = {}
for word in word_frequency:
    word_indexing[word] = word_index
    word_index +=1

with open(PICKLE_DATA_PATH + 'word_indexing', 'wb') as handle:
    pickle.dump(word_indexing, handle, protocol = pickle.HIGHEST_PROTOCOL)

data_helpers.get_tfidf_dic()
              

       
print("finished")
