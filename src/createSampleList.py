# import package
import json
import time
import datetime
import random
import numpy as np
import multiprocessing as mp
# import xgboost as xgb
import csv
from tqdm import tqdm  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances  
from sklearn.metrics.pairwise import manhattan_distances  

emb_file_dir='./emb/karate_512d.emb'
train_file_dir = './res/oriFile/train.csv'
test_file_dir = './res/oriFile/test.csv'
e2id_file_dir = './res/resFile/entity2id.txt'
r2id_file_dir = './res/resFile/relation2id.txt'
candidate_file_dir = './res/resFile/candidates.json'
sample_file_dir = './res/resFile/sample.csv'
counter_sample_file_dir = './res/resFile/counter_sample.csv'

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

def cal_sim(a_id,a_other_arr,emb_obj):
    sim_arr = []
    sim_func_arr = [euclidean_distances]
    sim_trans = [1]
    for idx,sim_func in enumerate(sim_func_arr):
        func_picked = idx
        sum = 0
        input_arr=[]
        for a_other_id in a_other_arr:
            input_arr.append(emb_obj[a_other_id])
        sum = sim_func([emb_obj[a_id]],input_arr)
        sim_arr.append(np.average(sum)*sim_trans[func_picked])
    
    return sim_arr
        
def create_counter_sample(b2a_obj,emb_obj):
    print ("start create_counter_sample")
    counter_sample_arr=[]
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
        # print sim_result
        sim_result = sim_result.T[0]
        ans_list = pick_3max(sim_result)
        ans_id_list = [data['candidates'][ans_list[0]],data['candidates'][ans_list[1]],data['candidates'][ans_list[2]]]
        for b_id in ans_id_list:
            for rel in b2a_obj[b_id].keys():
                cnt = 0
                for unwanted in b2a_obj[b_id][rel]:
                    random.shuffle(b2a_obj[b_id][rel])
                    if cnt == 3:
                        break
                    counter_sample_arr.append({
                        'a': a_id,
                        'b': unwanted,
                        'rel': rel
                    })
                    cnt = cnt + 1
    candidates_file_obj.close()
    print ("stop create_counter_sample")
    return counter_sample_arr

def build_sample_list(b2a_obj):
    sample_arr = []
    for b_key in b2a_obj.keys():
        b_id = b_key
        for rel in b2a_obj[b_key].keys():
            if len(b2a_obj[b_key][rel])>=6:
                random.shuffle(b2a_obj[b_key][rel])
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][1],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][2],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][4],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][5],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][2],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][4],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][5],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][2],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][2],
                    'b':b2a_obj[b_key][rel][4],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][2],
                    'b':b2a_obj[b_key][rel][5],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][3],
                    'b':b2a_obj[b_key][rel][4],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][3],
                    'b':b2a_obj[b_key][rel][5],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][4],
                    'b':b2a_obj[b_key][rel][5],
                    'rel': rel
                })
            elif len(b2a_obj[b_key][rel])>=4:
                random.shuffle(b2a_obj[b_key][rel])
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][1],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][2],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][2],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][1],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][2],
                    'b':b2a_obj[b_key][rel][3],
                    'rel': rel
                })
            elif len(b2a_obj[b_key][rel])>=2:
                random.shuffle(b2a_obj[b_key][rel])
                sample_arr.append({
                    'a':b2a_obj[b_key][rel][0],
                    'b':b2a_obj[b_key][rel][1],
                    'rel': rel
                })
    return sample_arr

# def build_counter_samper_list()

def write2file(sample_list,dir):
    fb = open(dir,"w")
    fb.write("a,b,rel\n")
    random.shuffle(sample_list)
    for item in sample_list:
        fb.write("%s,%s,%s\n"%(item['a'],item['b'],item['rel']))
    fb.close()


emb_obj = emb_reader()
e2id_id2e_obj = open_e2id_file()
id2e_obj = e2id_id2e_obj[1]
e2id_obj = e2id_id2e_obj[0]
r2id_obj = open_r2id_file()
b2a_obj = build_b2a_obj(e2id_obj,r2id_obj)
sample_arr = build_sample_list(b2a_obj)
write2file(sample_arr,sample_file_dir)
counter_sample_array = create_counter_sample(b2a_obj,emb_obj)
write2file(counter_sample_array,counter_sample_file_dir)
