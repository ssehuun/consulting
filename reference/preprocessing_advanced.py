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


def noun_grouping(tokenized_nouns):
    merged_nouns =[]
    # for only Mecab & Twitter function
    for noun_by_sentence in tokenized_nouns:
        merged_nouns_sentence = []
        merge_buffer = ['','']
        update_buffer = ''
        for noun in noun_by_sentence:
            if not noun == None:
                if (merge_buffer[1] != '' and merge_buffer[0] != ''):
                    merge_buffer[0] = str(merge_buffer[1])
                    merge_buffer[1] = str(noun[0])
                    merged_nouns_sentence.pop()
                elif (merge_buffer[0] == ''):
                    merge_buffer[0] = str(noun[0])
                elif (merge_buffer[1] == ''):
                    merge_buffer[1] = str(noun[0])
            else :
                if merge_buffer[0] != '' and merge_buffer[1] == '':
                    merged_nouns_sentence.append((merge_buffer[0], 'solo'))
                merge_buffer = ['', '']
                update_buffer = ''
            if (merge_buffer[0] != '' and merge_buffer[1] != ''):
                update_buffer = merge_buffer[0] + merge_buffer[1]
                merged_nouns_sentence.append((update_buffer, 'double'))
                merged_nouns_sentence.append((merge_buffer[0], 'solo_1'))
                merged_nouns_sentence.append((merge_buffer[1], 'solo_2'))
        merged_nouns.append(merged_nouns_sentence)
                    
    return merged_nouns

def grammar_check(tokenized_grammar, mod='Mecab'):
    processed_grammar = list(tokenized_grammar)
    processed_grammar_indentation = list(tokenized_grammar)
    for token_index in range(len(tokenized_grammar)):
        if (mod == 'Mecab'):
            if not('NNG' in tokenized_grammar[token_index][1] or 'NNP' in tokenized_grammar[token_index][1]):
                processed_grammar.remove(tokenized_grammar[token_index])
                processed_grammar_indentation[token_index] = None
        if (mod == 'Hannanum'):
            if (not 'N' in tokenized_grammar[token_index][1]):
                processed_grammar.remove(tokenized_grammar[token_index])
                processed_grammar_indentation[token_index] = None
        if (mod == 'Twitter'):
            if not ('Noun' in tokenized_grammar[token_index][1] or 'Suffix' in tokenized_grammar[token_index][1]):
                processed_grammar.remove(tokenized_grammar[token_index])
                processed_grammar_indentation[token_index] = None
    return processed_grammar, processed_grammar_indentation


            

def konlpy_tokenizing(document, mod = 'Mecab'):
    ##DOC FINAL RESULT INITIALIZATION
    tokenized_result = []
    noun_tokenized_result = []
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
        words, noun_words = grammar_check(words, mod)
        
        #result append
        tokenized_result.append(words)
        noun_tokenized_result.append(noun_words)
        
        
        # spacebar splits (data preparation for customized algorithm)
    processed_noun = noun_grouping(noun_tokenized_result)
    return real_document, tokenized_result, processed_noun
        

def tokenizing(document_path, analyzer = "Mecab"):
    print ("tokenizing function")
    with open(document_path, 'r', encoding="utf-8") as doc:
        real_doc, token, noun_token= konlpy_tokenizing(doc, analyzer)
    return real_doc, token, noun_token
    

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
    sample_doc, sample_token, noun_token = tokenizing(sample_data_path + '/' + sample_data[0], "Mecab")
    print ("####sample_doc####")
    print (sample_doc)
    print ()
    print ("####sample_token####")
    print (sample_token)
    
    #sample for Hannannum
    sample_doc, sample_token, noun_token = tokenizing(sample_data_path + '/' + sample_data[0], "Twitter")
    print ("####sample_doc####")
    print (sample_doc)
    print ()
    print ("####sample_token####")
    print (sample_token)
    print ()
    print ("####noun_token####")
    print (noun_token)
