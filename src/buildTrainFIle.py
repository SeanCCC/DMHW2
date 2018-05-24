import csv
import sys
from tqdm import tqdm

# dir define.
train_dir = './res/resFile/train.txt'
test_dir = './res/resFile/test.txt'
valid_dir = './res/resFile/valid.txt'
train_file_dir = './res/oriFile/train.csv'

def buildFiles4transE(input_arr):
    train_obj = open(train_dir,"w")
    test_obj = open(test_dir,"w")
    valid_obj = open(valid_dir,"w")    
    for item in tqdm(input_arr):
        train_obj.write("%s\t%s\t%s\n"%(item[0],item[2],item[1]))
    train_obj.close()

def open_train_file():
    train_file_obj = open(train_file_dir,"r")
    reader = csv.reader(train_file_obj)
    buildFiles4transE(reader)

open_train_file()
