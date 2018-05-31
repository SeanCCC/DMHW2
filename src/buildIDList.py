import csv
import sys
import random
from tqdm import tqdm

# dir define.
e2id_dir = './res/resFile/entity2id.txt'
r2id_dir = './res/resFile/relation2id.txt'
train_file_dir = './res/oriFile/train.csv'

def buildFiles4transE(input_arr):
    i=0
    e_pool = {}
    r_pool = {}
    e2id_obj = open(e2id_dir,"w")
    r2id_obj = open(r2id_dir,"w")
    e2id_obj.write("v,id\n")
    r2id_obj.write("v,id\n")
    for item in tqdm(input_arr):
        e_pool[item[0]]={}
        r_pool[item[1]]={}
        e_pool[item[2]]={}
        e_pool[item[0]]["exist"]=1
        r_pool[item[1]]["exist"]=1
        e_pool[item[2]]["exist"]=1
        if(item[1]=="field_is_part_of"):
            e_pool[item[0]]['role']='field'
            e_pool[item[2]]['role']='field'
        elif(item[1]=="author_is_in_field"):
            e_pool[item[0]]['role']='person'
            e_pool[item[2]]['role']='field'
        elif(item[1]=="paper_is_written_by"):
            e_pool[item[0]]['role']='paper'
            e_pool[item[2]]['role']='person'
        elif(item[1]=="work_in"):
            e_pool[item[0]]['role']='person'
            e_pool[item[2]]['role']='org'
        elif(item[1]=="paper_is_in_field"):
            e_pool[item[0]]['role']='paper'
            e_pool[item[2]]['role']='field'
        elif(item[1]=="paper_publish_on"):
            e_pool[item[0]]['role']='paper'
            e_pool[item[2]]['role']='journal'
        elif(item[1]=="paper_cit_paper"):
            e_pool[item[0]]['role']='paper'
            e_pool[item[2]]['role']='paper'
        
    for item in tqdm(e_pool.keys()):
        e2id_obj.write("%s,%d,%s\n"%(item,i,e_pool[item]['role']))
        i=i+1
    i=0
    for item in tqdm(r_pool.keys()):
        r2id_obj.write("%s,%d\n"%(item,i))
        i=i+1
    e2id_obj.close()
    r2id_obj.close()
    
        

def open_train_file():
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    csv_arr = []
    for item in tqdm(reader):
        csv_arr.append([item[0],item[1],item[2]])
    # random.shuffle(csv_arr)
    buildFiles4transE(csv_arr)
    train_file_obj.close()
open_train_file()
