# import package
import json
import time
import datetime
import random
import numpy as np
import xgboost as xgb
import csv
from tqdm import tqdm  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import classification_report

emb_file_dir='./emb/karate128d20nw.emb'
train_file_dir = './res/oriFile/train.csv'
test_file_dir = './res/oriFile/test.csv'
e2id_file_dir = './res/resFile/entity2id.txt'
r2id_file_dir = './res/resFile/relation2id.txt'
candidate_file_dir = './res/resFile/candidates.json'

def emb_reader():
    print "start reading emb"
    emb_file_obj = open(emb_file_dir,'r')
    emb_file_obj.readline()
    emb_obj={}
    for i in tqdm(range(17313)):
        tmp = emb_file_obj.readline().split()
        emb_obj[tmp[0]]=tmp[1:]
    print "stop reading emb"
    return emb_obj

def open_e2id_file():
    print "start reading e2id"
    e2id_file_obj = open(e2id_file_dir,"r")
    next(e2id_file_obj)
    reader = csv.reader(e2id_file_obj)
    e2id_obj={}
    id2e_obj={}
    for item in tqdm(reader):
        e2id_obj[item[0]]=item[1]
        id2e_obj[item[1]]=item[0]
    e2id_file_obj.close()
    print "stop reading e2id"
    return [e2id_obj,id2e_obj]

def open_r2id_file():
    print "start reading r2id"
    r2id_file_obj = open(r2id_file_dir,"r")
    next(r2id_file_obj)
    reader = csv.reader(r2id_file_obj)
    r2id_obj={}
    for item in tqdm(reader):
        r2id_obj[item[0]]=item[1]
    r2id_file_obj.close()
    print "stop reading r2id"
    return r2id_obj

def create_train_matrix(emb_obj,e2id_obj,r2id_obj):
    print "start create_train_matrix"
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    train_arr = []
    for item in tqdm(reader):
        train_arr.append([item[0],item[1],item[2]])
    # random.shuffle(train_arr)
    matrix = []
    label = []
    for item in tqdm(train_arr):
        a_id = e2id_obj[item[0]]
        b_id = e2id_obj[item[2]]
        label_now = r2id_obj[item[1]]
        tmp = emb_obj[a_id]+emb_obj[b_id]
        matrix.append(np.array(tmp).astype(np.float))
        label.append(int(label_now))
    matrix = np.asmatrix(matrix)
    result = {
        'train_matrix': matrix,
        'lable': label
    }
    print "stop create_train_matrix"
    return result

def train(param,num_round,train_mat):
    ## train
    print "start training"
    start_time = time.time()
    dtrain = xgb.DMatrix(train_mat['train_matrix'], label=train_mat['lable'])
    evallist  = [(dtrain,'train')]  # 這步可以不要，用於測試效果
    bst = xgb.train(param, dtrain, num_round, evallist)
    stop_time = time.time()
    print("## training done in %f seconds\n" % (stop_time-start_time))
    print "stop training"
    return bst

def pick_3max(arr):
    max = -10000
    idxa=-1
    idxb=-1
    idxc=-1
    for idx,item in enumerate(arr):
        if max < item:
            max = item
            idxa=idx
    max = -10000
    arr[idxa]=-10000
    for idx,item in enumerate(arr):
        if max < item:
            max = item
            idxb=idx
    max = -10000
    arr[idxb]=-10000
    for idx,item in enumerate(arr):
        if max < item:
            max = item
            idxc=idx
    max = -10000
    arr[idxc]=-10000
    return [idxa,idxb,idxc]

def testAndOutput(emb_obj,classifier,id2e_obj):
    print "start testing"
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_file_dir = "./result/result128d20nw.%s.csv" % (time_stamp)
    result_file_obj = open(result_file_dir,"w")
    result_file_obj.write("QueryId,ExpectedTail\n")
    candidates_file_obj = open(candidate_file_dir,"r")
    for i in tqdm(range(11529)): # 11529
        data_line = candidates_file_obj.readline()  
        data = json.loads(data_line)
        a_id = data['a']
        id = data['id']
        rel = int(data['rel'])
        test_mat = []
        # print data['candidates']
        for item in data['candidates']:
            b_id = item
            tmp = emb_obj[a_id]+emb_obj[b_id]
            test_mat.append(np.array(tmp).astype(np.float))   
        test_mat = np.asmatrix(test_mat)
        dtest = xgb.DMatrix(test_mat)
        predicted = classifier.predict(dtest)
        predicted = np.swapaxes(predicted,0,1)[rel]
        ans_list = pick_3max(predicted)
        ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
        # print ans_list
        result_file_obj.write("%s,%s %s %s\n"%(id,id2e_obj[ans_id_list[0]],id2e_obj[ans_id_list[1]],id2e_obj[ans_id_list[2]]))
    candidates_file_obj.close()
    print "stop testing"

emb_obj = emb_reader()
e2id_id2e_obj = open_e2id_file()
id2e_obj = e2id_id2e_obj[1]
e2id_obj = e2id_id2e_obj[0]
r2id_obj = open_r2id_file()
train_mat = create_train_matrix(emb_obj,e2id_obj,r2id_obj)
param = {'max_depth':10, 'eta':0.1, 'silent':1,'min_child_weight':10, 'subsample':0.8, 'objective':"multi:softprob", 'eval_metric':'merror', "colsample_bytree":0.5, 'num_class':7,'silent':1}
num_round = 200
classifier = train(param,num_round,train_mat)
testAndOutput(emb_obj,classifier,id2e_obj)
