import konlpy
import os
import re

DATA_PATH = 'sample_data'


def read_sample_data():
    sample_data_path = os.getcwd()
    sample_data_path = sample_data_path + '/' + DATA_PATH        
    sample_data = os.listdir(sample_data_path)
    return sample_data, sample_data_path

def re_substitute(sentence):
    sentence = re.sub(r"[0-9]+"," ", sentence)
    return sentence


def spacebar_based_grouping(sentence):
    # for only Mecab & Twitter function
    group = []
    spacebar_splits = [z for z in sentence.split(' ') if z]
    spacebar_array = [0]*len(spacebar_splits)
    return

def grammar_check(tokenized_grammar, mod='Mecab'):
    processed_grammar = list(tokenized_grammar)
    for token in tokenized_grammar:
        if (mod == 'Mecab'):
            if not('NNG' in token[1] or 'NNP' in token[1]):
                processed_grammar.remove(token)
        if (mod == 'Hannanum'):
            if (not 'N' in token[1]):
                processed_grammar.remove(token)
        if (mod == 'Twitter'):
            if not ('Noun' in token[1]):
                processed_grammar.remove(token)
    return processed_grammar


            

def konlpy_tokenizing(document, mod = 'Mecab'):
    ##DOC FINAL RESULT INITIALIZATION
    tokenized_result = []
    real_document = []
    
    for sentence in document:
        #replace enter key
        real_document.append(sentence.replace("\n",""))
        
        #substitute unnesessary keys 
        sentence = re_substitute(sentence)
        
        #konlpy korean 'Mecab'
        if mod =="Mecab":
            words = konlpy.tag.Mecab(dicpath='/usr/lib/mecab/dic/mecab-ko-dic').pos(sentence)
        elif mod =="Twitter":
            words = konlpy.tag.Twitter().pos(sentence)
        elif mod =="Hannanum":
            words = konlpy.tag.Hannanum().pos(sentence)
        elif mod == "Komoran":
            words = konlpy.tag.Komoran().pos(sentence)
        elif mod == "Kkma":
            words = konlpy.tag.Kkma().pos(sentence)
        
        #grammar selection
        words = grammar_check(words, mod)
        
        #result append
        tokenized_result.append(words)
        
        
        # spacebar splits (data preparation for customized algorithm)

    return real_document, tokenized_result
        

def tokenizing(document_path, analyzer = "Mecab"):
    print ("tokenizing function")
    with open(document_path, 'r', encoding="utf-8") as doc:
        real_doc, token= konlpy_tokenizing(doc, analyzer)
    return real_doc, token
    

if __name__ == '__main__':
    print ("this is tool for debugging preprocessing.py code")
    print ()
    print ("################################################")
    
    sample_data, sample_data_path = read_sample_data()
    print("sample_data list : ")
    print (sample_data)
    print ()
    print ("sample_data_path : ")
    print (sample_data_path)
    
    #####sample for just one document
    #sample for Mecab
    sample_doc, sample_token = tokenizing(sample_data_path + '/' + sample_data[0], "Mecab")
    print ("####sample_doc####")
    print (sample_doc)
    print ()
    print ("####sample_token####")
    print (sample_token)
    
    #sample for Hannannum
    sample_doc, sample_token = tokenizing(sample_data_path + '/' + sample_data[0], "Twitter")
    print ("####sample_doc####")
    print (sample_doc)
    print ()
    print ("####sample_token####")
    print (sample_token)
