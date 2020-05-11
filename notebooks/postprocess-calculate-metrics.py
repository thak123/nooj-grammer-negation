#!/usr/bin/env python
# coding: utf-8

import os


from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from allennlp.data.instance import Instance
from typing import Dict, List, Iterator
import itertools


# Convert the Nooj output to Conll format -> check which format is best => we used the annotated text format
# Read the Conan Doyale Negation Corpus
# pass the both the representations through grader
# for each sentence, prediction 
#   calculate the score and dispay the error analysis.
# repeat the same for the validation and test.


#PLACEHOLDERS

columns = {0: '#token', 1: 'negation_scope'}

GOLD_TRAIN_FILE_PATH = "/media/gaurish/angela/projects/nooj-grammer-negation/conandoyle_train.conllu"
PRED_TRAIN_FILE_PATH = "/media/gaurish/angela/projects/nooj-grammer-negation/565.txt"

GOLD_DEV_FILE_PATH = ""
PRED_DEV_FILE_PATH = ""

GOLD_TEST_FILE_PATH = ""
PRED_TEST_FILE_PATH = ""


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
    sentence_dict[sentence]=tag_tokens
    # print(">")

y_true = []
y_pred= []

with open(PRED_TRAIN_FILE_PATH) as input_file:
    for index,line in enumerate(input_file):
        line =line.strip()
        sentence ,tagged_sentence = (line.split("/"))
        tagged_sentence_tokens =tagged_sentence.split(">#<")[:-1]
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
                    # print(i.split(",")[1],"\t", scope_tag)
                    selected_text.append(i.split(",")[1])
                    
                else:
                    # print(i.split(",")[0],"\t", scope_tag)
                    selected_text.append(i.split(",")[0])
            selected_text_tags.append(scope_tag)
#         tagged_sentence = tagged_sentence.split("#",maxsplit=1)
#         print(tagged_sentence)
        # for token,tag in zip(selected_text,selected_text_tags) :
            # print(tag,token)
#         print(tagged_sentence)
        selected_text = " ".join(selected_text)
        # print(selected_text)
        len_st = len(selected_text)
        for tweet_index in (i for i, e in enumerate(sentence_dict.keys()) if selected_text in e):
            tweet = list(sentence_dict.keys())[tweet_index]
            ind = tweet.find(selected_text)
            text_tags = []
            if tweet[ind: ind+len_st] == selected_text:
                [text_tags.append("0") for i in tweet[0:ind].split()]
                [text_tags.append(i) for i in selected_text_tags]
                [text_tags.append("0") for i in tweet[ind+len_st:].split()]
                # print(selected_text_tags, tweet[0:ind].split(),tweet[ind+len_st:].split())
                if len(sentence_dict[tweet])== len(text_tags):
                    print("Text: ",tweet)
                    print(list(zip(tweet.split(), sentence_dict[tweet],text_tags)))
                
                    y_true.append(sentence_dict[tweet])
                    y_pred.append(text_tags)
                    print("F1", f1_score(sentence_dict[tweet], text_tags))
                    print("ACC", accuracy_score(sentence_dict[tweet], text_tags)) 
                    print("Report:", classification_report(sentence_dict[tweet], text_tags))
                    break
                else:
                    print("length not matching")
        # if index >5:  break
    print("Over >")


# In[136]:
print("Final Metric")
print(f1_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred)) 
print(classification_report(y_true, y_pred))









