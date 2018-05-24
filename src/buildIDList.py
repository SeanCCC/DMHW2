import csv
import sys
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
    for item in tqdm(input_arr):
        e_pool[item[0]]=1
        r_pool[item[1]]=1
        e_pool[item[2]]=1
    for item in tqdm(e_pool.keys()):
        e2id_obj.write("%s\t%d\n"%(item,i))
        i=i+1
    i=0
    for item in tqdm(r_pool.keys()):
        r2id_obj.write("%s\t%d\n"%(item,i))
        i=i+1
    e2id_obj.close()
    r2id_obj.close()
    
        

def open_train_file():
    train_file_obj = open(train_file_dir,"r")
    reader = csv.reader(train_file_obj)
    buildFiles4transE(reader)
    train_file_obj.close()
open_train_file()
