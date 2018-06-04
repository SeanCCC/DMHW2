# import package
import json
import time
import datetime
import random
import numpy as np
import multiprocessing as mp
import xgboost as xgb
import csv
from tqdm import tqdm  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances  
from sklearn.metrics.pairwise import manhattan_distances  

emb_file_dir='./emb/karate256d40nw5it.emb'
train_file_dir = './res/oriFile/train.csv'
test_file_dir = './res/oriFile/test.csv'
e2id_file_dir = './res/resFile/entity2id.txt'
r2id_file_dir = './res/resFile/relation2id.txt'
candidate_file_dir = './res/resFile/candidates.xgboost.json'
sample_file_dir = './res/resFile/sample.csv'
counter_sample_file_dir = './res/resFile/counter_sample.csv'

sim_func_arr = [ euclidean_distances,cosine_similarity,manhattan_distances]
sim_trans = [1,-1,-1]

# def testAndOutputXgboost(emb_obj,classifier,id2e_obj):
#     print "start testing"
#     time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
#     result_file_dir = "./result/result128d20nw.%s.csv" % (time_stamp)
#     result_file_obj = open(result_file_dir,"w")
#     result_file_obj.write("QueryId,ExpectedTail\n")
#     candidates_file_obj = open(candidate_file_dir,"r")
#     for i in tqdm(range(11529)): # 11529
#         data_line = candidates_file_obj.readline()  
#         data = json.loads(data_line)
#         a_id = data['a']
#         id = data['id']
#         rel = int(data['rel'])
#         test_mat = []
#         # print data['candidates']
#         for item in data['candidates']:
#             b_id = item
#             tmp = emb_obj[a_id]+emb_obj[b_id]
#             test_mat.append(np.array(tmp).astype(np.float))   
#         test_mat = np.asmatrix(test_mat)
#         dtest = xgb.DMatrix(test_mat)
#         predicted = classifier.predict(dtest)
#         predicted = np.swapaxes(predicted,0,1)[rel]
#         ans_list = pick_3max(predicted)
#         ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
#         # print ans_list
#         result_file_obj.write("%s,%s %s %s\n"%(id,id2e_obj[ans_id_list[0]],id2e_obj[ans_id_list[1]],id2e_obj[ans_id_list[2]]))
#     candidates_file_obj.close()
#     print "stop testing"

def trainXgboost(param,num_round,train_mat):
    ## train
    print "start training"
    start_time = time.time()
    dtrain = xgb.DMatrix(train_mat['train_matrix'], label=train_mat['lable'])
    evallist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    stop_time = time.time()
    print("## training done in %f seconds\n" % (stop_time-start_time))
    print "stop training"
    return bst

def create_train_matrix(emb_obj,e2id_obj,r2id_obj):
    print "start create_train_matrix"
    sample_file_obj = open(sample_file_dir,"r")
    next(sample_file_obj)
    reader = csv.reader(sample_file_obj)
    train_arr = []
    for item in tqdm(reader):
        train_arr.append([item[0],item[1],item[2]])
    matrix = []
    label = []
    for item in tqdm(train_arr):
        a_id = item[0]
        b_id = item[1]
        emb_a=emb_obj[a_id]
        emb_b=emb_obj[b_id]
        label_now = 1
        tmp=[]
        tmp.append(float(item[2]))
        for idx,func in enumerate(sim_func_arr):
            tmp.append(func([emb_a],[emb_b])[0][0]*sim_trans[idx])
        matrix.append(np.array(tmp).astype(np.float))
        label.append(label_now)
    ## counter
    counter_sample_file_obj = open(counter_sample_file_dir,"r")
    next(counter_sample_file_obj)
    reader = csv.reader(counter_sample_file_obj)
    train_arr = []
    for item in tqdm(reader):
        train_arr.append([item[0],item[1],item[2]])
    for item in tqdm(train_arr):
        a_id = item[0]
        b_id = item[1]
        emb_a=emb_obj[a_id]
        emb_b=emb_obj[b_id]
        label_now = 0
        tmp=[]
        tmp.append(float(item[2]))
        for idx,func in enumerate(sim_func_arr):
            tmp.append(func([emb_a],[emb_b])[0][0])
        matrix.append(np.array(tmp).astype(np.float))
        label.append(label_now)
    matrix = np.asmatrix(matrix)
    result = {
        'train_matrix': matrix,
        'lable': label
    }
    print "stop create_train_matrix"
    return result

def cosin_distance(vector1, vector2):  
    dot_product = 0.0  
    normA = 0.0  
    normB = 0.0  
    for a, b in zip(vector1, vector2):  
        dot_product += a * b  
        normA += a ** 2  
        normB += b ** 2  
    if normA == 0.0 or normB == 0.0:  
        return None  
    else:  
        return dot_product / ((normA * normB) ** 0.5)

def emb_reader():
    print ("start reading emb")
    emb_file_obj = open(emb_file_dir,'r')
    emb_file_obj.readline()
    emb_obj={}
    for i in tqdm(range(17313)):
        tmp = emb_file_obj.readline().split()
        emb_obj[tmp[0]]=np.array(tmp[1:]).astype(np.float)
    print ("stop reading emb")
    return emb_obj

def open_e2id_file():
    print ("start reading e2id")
    e2id_file_obj = open(e2id_file_dir,"r")
    next(e2id_file_obj)
    reader = csv.reader(e2id_file_obj)
    e2id_obj={}
    id2e_obj={}
    for item in tqdm(reader):
        e2id_obj[item[0]]=item[1]
        id2e_obj[item[1]]=item[0]
    e2id_file_obj.close()
    print ("stop reading e2id")
    return [e2id_obj,id2e_obj]

def open_r2id_file():
    print ("start reading r2id")
    r2id_file_obj = open(r2id_file_dir,"r")
    next(r2id_file_obj)
    reader = csv.reader(r2id_file_obj)
    r2id_obj={}
    for item in tqdm(reader):
        r2id_obj[item[0]]=item[1]
    r2id_file_obj.close()
    print ("stop reading r2id")
    return r2id_obj

def pick_3min(arr):
    min = 10000
    idxa=-1
    idxb=-1
    idxc=-1
    for idx,item in enumerate(arr):
        if min > item:
            min = item
            idxa=idx
    min = 10000
    arr[idxa]=10000
    for idx,item in enumerate(arr):
        if min > item:
            min = item
            idxb=idx
    min = 10000
    arr[idxb]=10000
    for idx,item in enumerate(arr):
        if min > item:
            min = item
            idxc=idx
    min = 10000
    arr[idxc]=10000
    return [idxa,idxb,idxc]


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

def build_b2a_obj(e2id_obj,r2id_obj):
    print ("start build_b2a_obj")
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    train_arr = []
    result_obj = {}
    for item in tqdm(reader):
        train_arr.append([item[0],item[1],item[2]])
    for item in tqdm(train_arr):
        b_id = e2id_obj[item[2]]
        r_id = r2id_obj[item[1]]
        a_id = e2id_obj[item[0]]
        if result_obj.get(b_id)==None:
            result_obj[b_id]={}
        if result_obj[b_id].get(r_id)==None:
            result_obj[b_id][r_id]=[]
        result_obj[b_id][r_id].append(a_id)
    print ("stop build_b2a_obj")
    train_file_obj.close()
    return result_obj

def trans_func(input):
    return input['func'](input['arg1'],input['arg2'])

def cal_sim_multiproc(a_id,a_other_arr,emb_obj):
    pool = mp.Pool()
    sim_arr = []
    sim_func_arr = [cosine_similarity]
    for sim_func in sim_func_arr:
        input_arr=[]
        for a_other_id in a_other_arr:
            input_arr.append({
                'arg1': emb_obj[a_id],
                'arg2': emb_obj[a_other_id],
                'func': sim_func
            })
        res = pool.map(trans_func, input_arr)
        max_arr=[]
        for i in range(5):
            max = np.max(res)
            res.remove(max)
            max_arr.append(max)
        avg = np.average(max_arr)
        sim_arr.append(avg)
    return sim_arr

def xgboost_sim(classifier,test_mat):
    test_mat = np.asmatrix(test_mat)
    dtest = xgb.DMatrix(test_mat)
    predicted = classifier.predict(dtest)
    return predicted

def cal_sim_xgboost(a_id,a_other_arr,emb_obj,classifier,rel):
    sim_arr = []
    f_rel = float(rel)
    sim_arr.append(f_rel)
    for idx,sim_func in enumerate(sim_func_arr):
        func_picked = idx
        sum = 0
        input_arr=[]
        for a_other_id in a_other_arr:
            input_arr.append(emb_obj[a_other_id])
        sum = sim_func([emb_obj[a_id]],input_arr)
        sim_arr.append(np.average(sum)*sim_trans[func_picked])
    predicted_arr = xgboost_sim(classifier,sim_arr)
    return predicted_arr

def cal_sim(a_id,a_other_arr,emb_objl):
    sim_arr = []
    sim_func_arr_t=sim_func_arr[:1]
    for idx,sim_func in enumerate(sim_func_arr_t):
        func_picked = idx
        sum = 0
        input_arr=[]
        for a_other_id in a_other_arr:
            input_arr.append(emb_obj[a_other_id])
        sum = sim_func([emb_obj[a_id]],input_arr)
        sim_arr.append(np.average(sum)*sim_trans[func_picked])
    return sim_arr
        
def compare_a2candi_output_xgboost(b2a_obj,emb_obj,classifier):
    print ("start compare_a2candi")
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file_dir = "./result/result256d40nw5iteucl.%s.csv" % (time_stamp)
    result_file_obj = open(result_file_dir,"w")
    result_file_obj.write("QueryId,ExpectedTail\n")
    candidates_file_obj = open(candidate_file_dir,"r")
    for i in tqdm(range(11529)): # 11529
        data_line = candidates_file_obj.readline()  
        data = json.loads(data_line)
        a_id = data['a']
        id = data['id']
        rel = data['rel']
        candi = data['candidates']
        sim_result=[]
        for b_id in candi:
            sim_arr = cal_sim_xgboost(a_id,b2a_obj[b_id][rel],emb_obj,classifier,rel)
            sim_result.append(sim_arr)
        sim_result=np.asarray(sim_result)
        # print sim_result
        # sim_result = sim_result.T[0]
        ans_list = pick_3max(sim_result)
        ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
        result_file_obj.write("%s,%s %s %s\n"%(id,id2e_obj[ans_id_list[0]],id2e_obj[ans_id_list[1]],id2e_obj[ans_id_list[2]]))
    candidates_file_obj.close()
    result_file_obj.close()
    print ("stop compare_a2candi with %s"%(result_file_dir))

def compare_a2candi_output(b2a_obj,emb_obj):
    print ("start compare_a2candi")
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file_dir = "./result/result512deucl.%s.csv" % (time_stamp)
    result_file_obj = open(result_file_dir,"w")
    result_file_obj.write("QueryId,ExpectedTail\n")
    candidates_file_obj = open(candidate_file_dir,"r")
    for i in tqdm(range(11529)): # 11529
        data_line = candidates_file_obj.readline()  
        data = json.loads(data_line)
        a_id = data['a']
        id = data['id']
        rel = data['rel']
        candi = data['candidates']
        sim_result=[]
        for b_id in candi:
            sim_arr = cal_sim(a_id,b2a_obj[b_id][rel],emb_obj)
            sim_result.append(sim_arr)
        sim_result=np.asarray(sim_result)
        print sim_result
        sim_result = sim_result.T[0]
        ans_list = pick_3max(sim_result)
        ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
        result_file_obj.write("%s,%s %s %s\n"%(id,id2e_obj[ans_id_list[0]],id2e_obj[ans_id_list[1]],id2e_obj[ans_id_list[2]]))
    candidates_file_obj.close()
    result_file_obj.close()
    print ("stop compare_a2candi with %s"%(result_file_dir))

def compare_a2candi_output_ht(b2a_obj,emb_obj):
    print ("start compare_a2candi")
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file_dir = "./result/result256d40nw5iteucl.%s.csv" % (time_stamp)
    result_file_obj = open(result_file_dir,"w")
    result_file_obj.write("QueryId,ExpectedTail\n")
    candidates_file_obj = open(candidate_file_dir,"r")
    for i in tqdm(range(11529)): # 11529
        data_line = candidates_file_obj.readline()  
        data = json.loads(data_line)
        a_id = data['a']
        id = data['id']
        rel = data['rel']
        candi = data['candidates']
        emb_list = []
        for b_id in candi:
            emb_list.append(emb_obj[b_id])
        sim_result=euclidean_distances(np.asarray([emb_obj[a_id]]),np.asarray(emb_list))[0]
        sim_result=np.asarray(sim_result)
        # print sim_result
        ans_list = pick_3min(sim_result)
        ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
        result_file_obj.write("%s,%s %s %s\n"%(id,id2e_obj[ans_id_list[0]],id2e_obj[ans_id_list[1]],id2e_obj[ans_id_list[2]]))
    candidates_file_obj.close()
    result_file_obj.close()
    print ("stop compare_a2candi with %s"%(result_file_dir))

emb_obj = emb_reader()
e2id_id2e_obj = open_e2id_file()
id2e_obj = e2id_id2e_obj[1]
e2id_obj = e2id_id2e_obj[0]
r2id_obj = open_r2id_file()
b2a_obj = build_b2a_obj(e2id_obj,r2id_obj)
# train_mat = create_train_matrix(emb_obj,e2id_obj,r2id_obj)
# param = {'max_depth':10, 'eta':0.1, 'silent':1,'min_child_weight':11, 'subsample':0.8, 'objective':"binary:logistic", 'eval_metric':'auc', "colsample_bytree":0.5}
# num_round = 200
# classifier = trainXgboost(param,num_round,train_mat)
compare_a2candi_output_ht(b2a_obj,emb_obj)
