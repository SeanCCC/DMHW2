import csv
import sys
import json
from tqdm import tqdm

# dir define.
e2id_file_dir = './res/resFile/entity2id.txt'
test_file_dir = './res/oriFile/test.csv'
candidate_file_dir = './res/resFile/cadidates.json'

def open_e2id_file():
    e2id_file_obj = open(e2id_file_dir,"r")
    next(e2id_file_obj)
    reader = csv.reader(e2id_file_obj)
    e2id_obj={}
    role_obj={}
    role_obj['person']=[]
    role_obj['org']=[]
    role_obj['paper']=[]
    role_obj['field']=[]
    role_obj['journal']=[]
    for item in tqdm(reader):
        role_obj[item[2]].append(item[1])
        e2id_obj[item[0]]=item[1]
    e2id_file_obj.close()
    result={
        'role':role_obj,
        'e2id':e2id_obj
    }
    return result

def open_test_file(role_e2id_obj):
    map2role={
        'field_is_part_of':'field',
        'author_is_in_field':'field',
        'paper_is_written_by':'person',
        'work_in':'org',
        'paper_is_in_field':'field',
        'paper_publish_on':'journal',
        'paper_cit_paper':'paper'
    }
    test_file_obj = open(test_file_dir,"r")
    fd = open(candidate_file_dir,'w')  
    next(test_file_obj)
    reader = csv.reader(test_file_obj)
    for item in tqdm(reader):
        tmp_json={
            'id':item[0],
            'a':role_e2id_obj['e2id'][item[1]],
            'candidates':role_e2id_obj['role'][map2role[item[2]]]
        }
        fd.write(json.dumps(tmp_json, separators=(',',':'))+"\n")
    fd.close()
    test_file_obj.close()

role_e2id_obj = open_e2id_file()
open_test_file(role_e2id_obj)

