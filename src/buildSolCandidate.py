import csv
import sys
import json
import copy
from tqdm import tqdm

# dir define.
e2id_file_dir = './res/resFile/entity2id.txt'
r2id_file_dir = './res/resFile/relation2id.txt'
test_file_dir = './res/oriFile/test.csv'
train_file_dir = './res/oriFile/train.csv'
candidate_file_dir = './res/resFile/candidates.json'

def open_r2id_file():
    r2id_file_obj = open(r2id_file_dir,"r")
    next(r2id_file_obj)
    reader = csv.reader(r2id_file_obj)
    r2id_obj={}
    for item in tqdm(reader):
        r2id_obj[item[0]]=item[1]
    r2id_file_obj.close()
    return r2id_obj

def open_e2id_file():
    e2id_file_obj = open(e2id_file_dir,"r")
    next(e2id_file_obj)
    reader = csv.reader(e2id_file_obj)
    e2id_obj={}
    for item in tqdm(reader):
        e2id_obj[item[0]]=item[1]
    e2id_file_obj.close()
    result={
        'e2id':e2id_obj
    }
    return result

def open_train_file(role_e2id_obj,r2id):
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    train_set = {}
    e2id=role_e2id_obj['e2id']
    for item in tqdm(reader):
        a_id = e2id[item[0]]
        if train_set.get(a_id)==None:
            train_set[a_id]={}
        if train_set[a_id].get(r2id[item[1]])==None:
            train_set[a_id][r2id[item[1]]]=[]
        train_set[a_id][r2id[item[1]]].append(e2id[item[2]])
    train_file_obj.close()
    return train_set

def open_train_file4rel(role_e2id_obj,r2id):
    train_file_obj = open(train_file_dir,"r")
    next(train_file_obj)
    reader = csv.reader(train_file_obj)
    e2id=role_e2id_obj['e2id']
    rel_obj={}
    for item in tqdm(reader):
        if rel_obj.get(item[1]) == None:
            rel_obj[item[1]] = []
        if e2id[item[2]] not in rel_obj[item[1]]:
            rel_obj[item[1]].append(e2id[item[2]])
    train_file_obj.close()
    return rel_obj

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def open_test_file(role_e2id_obj,r2id_obj,train_set,rel_obj):
    test_file_obj = open(test_file_dir,"r")
    fd = open(candidate_file_dir,'w')  
    next(test_file_obj)
    reader = csv.reader(test_file_obj)
    for item in tqdm(reader):
        a_id=role_e2id_obj['e2id'][item[1]]
        relation = r2id_obj[item[2]]
        # building candidates
        candi = rel_obj[item[2]]
        overlap=[]
        if(train_set.get(a_id)!=None) and (train_set[a_id].get(relation)!=None):
            overlap = intersection(candi,train_set[a_id][relation])
        if(a_id in candi) or (len(overlap)!=0):
            candi = copy.deepcopy(rel_obj[item[2]])
        if(len(overlap)!=0):
            for same in overlap:
                candi.remove(same)
        if(a_id in candi):
            overlap_idx = candi.index(a_id)
            del(candi[overlap_idx])

        # write to file
        tmp_json={
            'id':item[0],
            'a':a_id,
            'rel': relation,
            'candidates':candi
        }
        fd.write(json.dumps(tmp_json, separators=(',',':'))+"\n")
    fd.close()
    test_file_obj.close()

role_e2id_obj = open_e2id_file()
r2id_obj = open_r2id_file()
rel_obj = open_train_file4rel(role_e2id_obj,r2id_obj)
train_set = open_train_file(role_e2id_obj,r2id_obj)
open_test_file(role_e2id_obj,r2id_obj,train_set,rel_obj)
