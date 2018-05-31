import csv
import sys
import random
from tqdm import tqdm
from numpy import genfromtxt

# dir define.
train_transE_dir = './res/resFile/transE/train.txt'
test_transE_dir = './res/resFile/transE/test.txt'
valid_transE_dir = './res/resFile/transE/valid.txt'
train_node2vec_dir = './res/resFile/node2vec/train.edgelist'
train_file_dir = './res/oriFile/train.csv'

def open_e_list_file():
    e_list_dir = './res/resFile/entity2id.txt'
    e_list_arr = {}
    e_list_dir = open(e_list_dir,"r")
    next(e_list_dir)
    reader = csv.reader(e_list_dir)
    for item in tqdm(reader):
        e_list_arr[item[0]]=item[1]
    return e_list_arr

def buildFiles4transE(input_arr):
    train_obj = open(train_transE_dir,"w")
    test_obj = open(test_transE_dir,"w")
    valid_obj = open(valid_transE_dir,"w")    
    for item in tqdm(input_arr):
        train_obj.write("%s\t%s\t%s\n"%(item[0],item[2],item[1]))
    train_obj.close()

def buildFiles4Node2Vec(input_arr,e_list):
    train_obj = open(train_node2vec_dir,"w")
    for item in tqdm(input_arr):
        train_obj.write("%s %s\n"%(e_list[item[0]],e_list[item[2]]))
    train_obj.close()

def open_train_file(e_list):
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    csv_arr = []
    for item in tqdm(reader):
        csv_arr.append([item[0],item[1],item[2]])
    # random.shuffle(csv_arr)
    # buildFiles4transE(csv_arr)
    buildFiles4Node2Vec(csv_arr,e_list)

e_list = open_e_list_file()
open_train_file(e_list)
