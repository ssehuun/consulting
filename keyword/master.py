import os
import subprocess
import gevent
import multiprocessing

cwd = os.getcwd()
def command1 ():
    result = subprocess.check_output('python3 reduce_token_1.py')
def command2 (i):
    result = subprocess.check_output('python3 ~/SK/parse/reduce_token_2_v2.py ~/SK/parse/document_dict' + str(i), shell=True)
    print ("here")
def command3():
    result = subprocess.check_output('python3 reduce_token_3.py')
    result = subprocess.check_output('python3 make_tfidf_dic.py')

#print("1st")
#gevent.joinall([
#        gevent.spawn(command1())
#        ])

print("2nd")
pool = multiprocessing.Pool(processes=4)

pool.map(command2, range(1,5))
#threads = [gevent.spawn(command2,i) for i in range(1,5)]
#gevent.joinall(threads)

#print ("3rd")
#gevent.joinall([
#        gevent.spanw(command3())
#        ])
#print ("final")
#result = subprocess.check_output('python3 finalize_token.py')
print("finished")