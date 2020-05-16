#!/usr/bin/env python
# coding: utf-8
import os


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



len(target_lines),len(source_lines)
current_i_index =0
current_j_index=0
#loop through the multiple parts
for i_index, i in enumerate(source_lines):
    # check if the part 
    for j_index , j in enumerate(target_lines[current_j_index:]):
        sent,_ = j.split("/")
        if sent in i:
            print(i)
            current_j_index +=1
        else:
            break

    # if j_index>5: break



