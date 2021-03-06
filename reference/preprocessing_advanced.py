import konlpy
import os
import re

## data directory path starting from this directory
DATA_PATH = 'sample_data'



## reading and testing sample data
def read_sample_data():
    sample_data_path = os.getcwd()
    sample_data_path = sample_data_path + '/' + DATA_PATH        
    sample_data = os.listdir(sample_data_path)
    return sample_data, sample_data_path


## you can substitute specific string by regular expression 
def re_substitute(sentence):
    sentence = re.sub(r"[0-9]+"," ", sentence)
    return sentence


## you can substitute specific token by inserting your own token by replacing or adding into 'something'
def token_substitute(tokenized_doc):
    processed_doc = list(tokenized_doc)
    for tokenized_sentence in tokenized_doc:
        for token in tokenized_sentence:
            if token == None:
                continue
            if ('something' in token[0]):
                processed_doc.remove(token)
    return processed_doc


##re-group the token which is too narrowly divided
## this function is mainly focused on Twitter analyzer
def token_grouping(tokenized_nouns):
    merged_nouns =[]
    # for only Mecab & Twitter function
    for noun_by_sentence in tokenized_nouns:
        merged_nouns_sentence = []
        merge_buffer = ['','']
        update_buffer = ''
        
        ## initialization for Twitter_verb
        Twitter_verb = False
        verb_update_buffer = ''
        
        for noun in noun_by_sentence:
            ## only for Twitter ##
            
            if not noun == None and noun[1] == 'Josa':
                continue
            
            ### add solo Adverb ###
            if not noun == None and noun[1] =='Adverb':
                merged_nouns_sentence.append(noun)
                continue
            
            #### 'Adjective or Verb merging ###
            
            if not noun == None and (noun[1] == 'Adjective' or noun[1] == 'Verb'):
                if (Twitter_verb == True):
                    merged_nouns_sentence.append((verb_update_buffer, 'verb_adjective'))
                if (merge_buffer[0] != '' and merge_buffer[1] == ''):
                    merged_nouns_sentence.append((merge_buffer[0], 'solo'))
                    merge_buffer = ['','']
                    update_buffer = []
                if (merge_buffer[0] != '' and merge_buffer[1] != ''):
                    merge_buffer = ['','']                    
                    update_buffer = []
                Twitter_verb = True
                verb_update_buffer = noun[0]
                continue
            
            if not noun == None and (noun[1] =='Eomi' or noun[1] =='PreEomi'):
                if (Twitter_verb == True):
                    verb_update_buffer += noun[0]
                continue
                
            if Twitter_verb == True:
                merged_nouns_sentence.append((verb_update_buffer, 'verb_adjective'))
                verb_update_buffer = ''
                Twitter_verb =False
                
                
                
            ## only for Twitter ends ##    
                
                
            ### 'NOUN' merging ###
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
        if (verb_update_buffer != ''):
            merged_nouns_sentence.append((verb_update_buffer, 'verb_adjective'))
        merged_nouns.append(merged_nouns_sentence)
                    
    return merged_nouns


## grammar_check for konlpy tokenized document
## processed_grammar_indentation : replace unnecessary grammar to None
def grammar_check(tokenized_grammar, mod='Mecab'):
    processed_grammar_indentation = list(tokenized_grammar)
    for token_index in range(len(tokenized_grammar)):
        if (mod == 'Mecab'):
            if not('NNG' in tokenized_grammar[token_index][1] or 'NNP' in tokenized_grammar[token_index][1]):
                processed_grammar_indentation[token_index] = None
        if (mod == 'Hannanum'):
            if (not 'N' in tokenized_grammar[token_index][1]):
                processed_grammar_indentation[token_index] = None
        if (mod == 'Twitter'):
            if not ('Noun' in tokenized_grammar[token_index][1] or 'Suffix' in tokenized_grammar[token_index][1] or 'Josa' in tokenized_grammar[token_index][1] or 'Adverb' in tokenized_grammar[token_index][1] \
                    or 'Adjective' in tokenized_grammar[token_index][1] or 'Verb' in tokenized_grammar[token_index][1] or 'Eomi' in tokenized_grammar[token_index][1] or 'PreEomi' in tokenized_grammar[token_index][1]):
                processed_grammar_indentation[token_index] = None
    return processed_grammar_indentation


            
## tokenizing and analyzing received document by specific method
## such as konlpy , grammar_check(tokenized_grammar, mod), token_grouping(tokenized_nouns), token_substitute(tokenized_doc), re_substitute(sentence)
## return real_document, tokenized_result, processed_result
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
        tokenized_result.append(words)
        indented_words = grammar_check(words, mod)
        
        #result append
        
        noun_tokenized_result.append(indented_words)
        
        
    preprocessed_result = token_substitute(noun_tokenized_result)
    processed_result = token_grouping(preprocessed_result)
    return real_document, tokenized_result, processed_result
        

    
## need to be revised ## (original purpose : run all of document in document path)
## now : run/anlayze specific document passed by 'document_path'
def tokenizing(document_path, analyzer = "Mecab"):
    print ("tokenizing function")
    with open(document_path, 'r', encoding="utf-8") as doc:
        real_doc, token, noun_token= konlpy_tokenizing(doc, analyzer)
    return real_doc, token, noun_token
    

### if__name__ == '__main__' is only for running script directly ex) python3 preprocessing_advanced.py
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
