import konlpy
import os
import re
import subprocess
#should be finalized as "import preprocessing"
import preprocessing_advanced as preprocessing

DATA_PATH = 'sample_data'

def sub_process_preprocessing(process_num):
    #preprocessing_advanced.py should be changed to recognize batch_num
    result = subprocess.check_output('python3 preprocessing_advanced')

if __name__ == '__main_':
    print ("this is tool for debugging master.py code")
    print ()
    print ("##########################################")
    
    #read_sample_data could be relpaced to data_helpers.py , to-be
    sample_data, sample_data_path = preprocessing.read_sample_data()
    
    
