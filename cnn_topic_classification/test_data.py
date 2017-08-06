import numpy as np
import os
import time
import pickle
import sys
import pandas as pd	
import argparse
import re
import itertools
from collections import Counter


if __name__ == "__main__":

    test_df = pd.read_csv("test.csv")
    test_q1 = test_df.question1.astype(str)
    test_q2 = test_df.question2.astype(str)

    word_index_pickle = open('test_data_pickle', 'wb')
    pickling = {'question1': test_q1, 'question2': test_q2}
    pickle.dump(pickling, word_index_pickle)


