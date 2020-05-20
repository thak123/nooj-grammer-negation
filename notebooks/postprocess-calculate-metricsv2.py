#!/usr/bin/env python
# coding: utf-8

import os
import sys

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from allennlp.data.instance import Instance
from typing import Dict, List, Iterator
import itertools

import re

columns = {0: '#token', 1: 'negation_scope'}
OTHER_TAG = "O"
GOLD_TRAIN_FILE_PATH = "/media/gaurish/angela/projects/nooj-grammer-negation/conandoyle_train.conllu"
PRED_TRAIN_FILE_PATH = "/media/gaurish/angela/projects/nooj-grammer-negation/565v3.txt"
SOURCE_FILE_PATH = "notebooks/conandoyle_train.html"


GOLD_DEV_FILE_PATH = ""
PRED_DEV_FILE_PATH = ""

GOLD_TEST_FILE_PATH = ""
PRED_TEST_FILE_PATH = ""

source_lines =[]
with open(SOURCE_FILE_PATH) as source_file:
    for line in source_file:
        line= line.strip()
        if line:
            source_lines.append(line)


def _read(file_path: str =GOLD_TRAIN_FILE_PATH) -> Iterator[Instance]:
    is_divider = lambda line: line.strip() == '' or len(line.strip().split())> 2 or line.strip()=="#token	negation_scope" 
    with open(file_path, 'r') as conll_file:
        for divider, lines in itertools.groupby(conll_file, is_divider):
            if not divider:
                fields = [l.strip().split() for l in lines]
                # switch it so that each field is a list of tokens/labels
                fields = [l for l in zip(*fields)]
                # only keep the tokens and NER labels
                tokens, ner_tags = fields
                yield zip(tokens, ner_tags)

#This is GOLD sentence dict
sentence_dict= {}

for zipped_tokens in _read():
    sentence_tokens = []
    tag_tokens = []
    for zipped_token in zipped_tokens:
        token, tag = zipped_token
        sentence_tokens.append(token)
        tag_tokens.append(tag.replace("_","-"))
    sentence = " ".join(sentence_tokens)
    if sentence in sentence_dict:
        print("found same entity")
    else:
        sentence_dict[sentence]=tag_tokens
    # print(">")

y_true = []
y_pred= []

current_j_index=0

from collections import defaultdict 
red_sentence_dict= defaultdict(list) 

target_lines = []
with open(PRED_TRAIN_FILE_PATH) as target_file:
    for line in target_file:
        line= line.strip()
        if line:
            target_lines.append(line)

for i_index, i in enumerate(source_lines):
    #loop through the multiple parts
    for j_index , j in enumerate(target_lines[current_j_index:]):
        sent,tagged_sentence = j.split("/")
        # check if the part 
        if sent in i:
            print(i,"=",sent)
            current_j_index +=1
            red_sentence_dict[i].append(tagged_sentence)
        else:
            break

def return_word_tag(tagged_sentence):
    tagged_sentence = tagged_sentence.replace("<NEG-CUE#<n,,WF>#<''',,WF>#<t,,WF>#>","<NEG-CUE#<n't,,WF>#>")
    tagged_sentence = tagged_sentence.replace("<'-',,WF>#<'-',,WF>","<'-',,WF>")
    tagged_sentence = tagged_sentence.replace("<NEG-CUE#<>#","<NEG-CUE#")
    if ">#<" in tagged_sentence:
        tagged_sentence_tokens =tagged_sentence.split(">#<")[:-1]
    elif "NEG-CUE#<" in tagged_sentence:
        tagged_sentence_tokens =[tagged_sentence[1:-2]]

    #logic for merging multi-word expressions:
    tst = copy.deepcopy(tagged_sentence_tokens)
    skiped_index = -1
    for e_index, element in enumerate(tst):
        if "FE" == element:
            tst.pop(e_index)
            skiped_index = e_index
            
        elif skiped_index != -1:
            c =tst.pop(skiped_index).split(",")
            m= tst.pop(skiped_index).split(",")
            r= tst.pop(skiped_index).split(",")
            tmp =  ",".join([c[0]+"-"+r[0],c[0]+"-"+r[0],"WF"])
            tst.insert(skiped_index, tmp)
            skiped_index=-1
    
    tagged_sentence_tokens =tst

    selected_text = []
    selected_text_tags =[]
    full_text = []
    previous_negated = False
    previous_first_negated =False
    for t_index, i in enumerate(tagged_sentence_tokens):
        scope_tag = ""
        
        #check if negation or scope    
        if "NEG-CUE#" in i:
            negated_token =i.split(",")[0].replace("NEG-CUE#<","")
            # print(negated_token, "\t", "B-NEG")
            if  "<" in negated_token:
                negated_token = negated_token.replace("<","")

            selected_text.append(negated_token)
            if not previous_first_negated:
                scope_tag = "B-cue" 
                previous_first_negated = True
            else:
                scope_tag = "I-cue"
                previous_first_negated = False

            previous_negated =True
        else:
            # scope
            # not negated/scope and 
            if t_index ==0 or previous_negated == True:
                scope_tag = "B-scope"
                previous_negated = False
            else:
                scope_tag = "I-scope"
            if  "<" in i:
                i = i.replace("<","")
                selected_text.append(i.split(",")[1])
                
            else:
                selected_text.append(i.split(",")[0])
        selected_text_tags.append(scope_tag)
    return selected_text,selected_text_tags
import copy
counter = 0
for red_sent in list(red_sentence_dict.keys()):
    key = red_sent.replace("<red>","").replace("</red>","")
    gold_tags = sentence_dict[key]
    pred_tags_collection = red_sentence_dict[red_sent]
    counter +=1
    # if counter ==449 :
        # g = input("Enter your name : ") 
        # pass
    print(gold_tags,key.split(),counter)
    blank_tags = [OTHER_TAG]*len(key.split())
    last_index =0
    text_tags_collection=[]
    for pred_tags in pred_tags_collection:
        selected_text, selected_text_tags = return_word_tag(pred_tags)
        # need to put this tags in the gold tags
        selected_text = " ".join(selected_text)
        len_st = len(selected_text)
        tweet = key
        ind = tweet.find(selected_text,last_index)
        text_tags = []
        if tweet[ind: ind+len_st] == selected_text:
            last_index  =ind
            if selected_text.split()[0] in ['s','ve','ll','m','d','re','n'] and tweet[ind-1]=="'":
                [text_tags.append(OTHER_TAG) for i in tweet[0:ind-1].split()]
                [text_tags.append(i) for i in selected_text_tags]
                [text_tags.append(OTHER_TAG) for i in tweet[ind+len_st:].split()]
            elif selected_text.split()[0] in ['t'] and tweet[ind-2:ind]=="n'":
                [text_tags.append(OTHER_TAG) for i in tweet[0:ind-2].split()]
                [text_tags.append(i) for i in selected_text_tags]
                [text_tags.append(OTHER_TAG) for i in tweet[ind+len_st:].split()]
            elif selected_text.split()[-1] in ['L',"Mrs"] and tweet[ind+ind+len_st:ind+ind+len_st+1]==".":
                [text_tags.append(OTHER_TAG) for i in tweet[0:ind].split()]
                [text_tags.append(i) for i in selected_text_tags]
                [text_tags.append(OTHER_TAG) for i in tweet[ind+1+len_st:].split()]
            else:
                [text_tags.append(OTHER_TAG) for i in tweet[0:ind].split()]
                [text_tags.append(i) for i in selected_text_tags]
                [text_tags.append(OTHER_TAG) for i in tweet[ind+len_st:].split()]
            text_tags_collection.append(text_tags)
        print()

    final_tag = copy.deepcopy(text_tags_collection[0])
    for t_index, t in enumerate(text_tags_collection[0]):
        prev_value  = OTHER_TAG
        if len(text_tags_collection)>1:
            for i_index in range(1, len(text_tags_collection)):
                if text_tags_collection[i_index][t_index] != OTHER_TAG:
                    final_tag[t_index]= text_tags_collection[i_index][t_index]
    print(final_tag)
    if len(sentence_dict[key]) != len(final_tag):
        print("not fine")
        sys.exit()
    else:
        for i in list(zip(tweet.split(), sentence_dict[tweet],final_tag)):
            print(i)
        y_true.append(sentence_dict[tweet])
        y_pred.append(final_tag)
        print("F1", f1_score(sentence_dict[tweet], final_tag))
        print("ACC", accuracy_score(sentence_dict[tweet], final_tag)) 
        print("Report:", classification_report(sentence_dict[tweet], final_tag))

# with open(PRED_TRAIN_FILE_PATH) as input_file:
#     last_index = 0
#     for index,line in enumerate(input_file):
#         line =line.strip()
#         sentence ,tagged_sentence = (line.split("/"))
#         if ">#<" in tagged_sentence:
#             tagged_sentence_tokens =tagged_sentence.split(">#<")[:-1]
#         elif "NEG-CUE#<" in tagged_sentence:
#             tagged_sentence_tokens =[tagged_sentence[1:-2]]
#         selected_text = []
#         selected_text_tags =[]
#         full_text = []
#         previous_negated = False
#         previous_first_negated =False
#         for t_index, i in enumerate(tagged_sentence_tokens):
#             scope_tag = ""
            
#             #check if negation or scope    
#             if "NEG-CUE#" in i:
#                 negated_token =i.split(",")[0].replace("NEG-CUE#<","")
#                 # print(negated_token, "\t", "B-NEG")
#                 if  "<" in negated_token:
#                     negated_token = negated_token.replace("<","")

#                 selected_text.append(negated_token)
#                 if not previous_first_negated:
#                     scope_tag = "B-cue" 
#                     previous_first_negated = True
#                 else:
#                     scope_tag = "I-cue"
#                     previous_first_negated = False

#                 previous_negated =True
#             else:
#                 # scope
#                 # not negated/scope and 
#                 if t_index ==0 or previous_negated == True:
#                     scope_tag = "B-scope"
#                     previous_negated = False
#                 else:
#                     scope_tag = "I-scope"
#                 if  "<" in i:
#                     i = i.replace("<","")
#                     selected_text.append(i.split(",")[1])
                    
#                 else:
#                     selected_text.append(i.split(",")[0])
#             selected_text_tags.append(scope_tag)
#         selected_text = " ".join(selected_text)
#         len_st = len(selected_text)
#         sample_sent_dict = list(sentence_dict.keys())
#         for tweet_index in (i for i, e in enumerate(sample_sent_dict) if selected_text in e):
#             tweet = sample_sent_dict[tweet_index]
#             ind = tweet.find(selected_text)
#             text_tags = []
#             if tweet[ind: ind+len_st] == selected_text:
#                 last_index = tweet_index
#                 [text_tags.append(OTHER_TAG) for i in tweet[0:ind].split()]
#                 [text_tags.append(i) for i in selected_text_tags]
#                 [text_tags.append(OTHER_TAG) for i in tweet[ind+len_st:].split()]
#                 if len(sentence_dict[tweet])== len(text_tags):
#                     print("Text: ",tweet)
#                     for i in list(zip(tweet.split(), sentence_dict[tweet],text_tags)):
#                         print(i)
#                     y_true.append(sentence_dict[tweet])
#                     y_pred.append(text_tags)
#                     print("F1", f1_score(sentence_dict[tweet], text_tags))
#                     print("ACC", accuracy_score(sentence_dict[tweet], text_tags)) 
#                     print("Report:", classification_report(sentence_dict[tweet], text_tags))
#                     break
#                 else:
#                     print("length not matching")
#             else:
#                 print("Something happened")
#         # if index >5:  break
#     print("Over >")


# In[136]:
print("Final Metric")
print(f1_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred)) 
print(classification_report(y_true, y_pred))










