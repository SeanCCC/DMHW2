import csv
import sys

# dir define.
train_file_dir = './res/oriFile/train.csv'
train_file_obj = open(train_file_dir,"r")
reader = csv.reader(train_file_obj)
for item in reader:
    print(item[2])