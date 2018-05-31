# import package
import json
import time
import datetime
import numpy as np
import xgboost as xgb

from tqdm import tqdm  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import classification_report

emb_file_dir='./emb/karate.emb'

def emb_reader():
    emb_file_obj = open(emb_file_dir,'r')
    emb_file_obj.readline()
    emb_obj={}
    for i in tqdm(range(17313)):
        tmp = emb_file_obj.readline().split()
        emb_obj[tmp[0]]=tmp[1:]
    return emb_obj

emb_obj = emb_reader()
    