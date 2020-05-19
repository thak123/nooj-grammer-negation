#!/usr/bin/env python
# coding: utf-8
import os

lr_context = "../565vlr.txt"
context = "../565v3.txt"
colored_file = "../conandoyle_train.rtf"


target_lines = []
with open("565v3.txt") as target_file:
    for line in target_file:
        line= line.strip()
        if line:
            target_lines.append(line)


source_lines =[]
with open("notebooks/conandoyle_train.html") as source_file:
    for line in source_file:
        line= line.strip()
        if line:
            source_lines.append(line)



print(len(target_lines),len(source_lines))
current_j_index=0

from collections import defaultdict 
red_sentence_dict= defaultdict(list) 

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

    # if j_index>5: break



