import os
import json
import random
from multiprocessing import Pool
from io import open

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import argparse
from sklearn import preprocessing



from utils import User
import random



train_ratio = 0.8
max_len_cas =100
cas_lens = [50,40,30,20,10]
pred_len = 100

text_model_addr = 'd2v/weibo_d2v.model'





def make_all_users(data_addr,data_class_addr, users_addr):
    data_dir = data_addr
    engaged_users={}
    user_id ={}
    labels = {}
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = int(line[line.index('eid:') + 4: line.index('label')])
            label = int(line[line.index('label:') + 6])
            labels[eid] = label

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            eid = int(filename.split('.')[0])
            label = labels[eid]
            if label == 1:
                with open(data_addr + '/' + filename, encoding="utf8") as f:
                    event_data = json.load(f)
                    for tweet in event_data:
                        user = User(tweet)
                        if user.id in engaged_users:
                            engaged_users[user.id]+=1
                        else:
                            engaged_users[user.id]=1
                            user_id[user.id] = user

    print('before all users')
#    print(engaged_users)
    all_users = list(set([user_id[userid] for userid,count in engaged_users.items() if count>0]))

    print('after all usesrs')
    user_embedding =[]
    for user in all_users:
        user_embedding.append(user.vector[0:5])
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(user_embedding)
    user_embedding = scaler.transform(user_embedding)
    print('before embedding')

    with open(users_addr, 'wt') as users_f:
        for i, user in enumerate(all_users):
            new_user_vector = user.vector.tolist()
            # print('before' , new_user_vector)
            new_user_vector[0:5] = user_embedding[i]
            # print('after' , new_user_vector ,user_embedding[i])
            user_id = user.id
            json_data = json.dumps([new_user_vector , int(user_id)])
            users_f.write(json_data + '\n')


def make_user_dic(user_addr):
    user_data = [json.loads(d) for d in open(user_addr, "rt").readlines()]
    user_dic={}
    for user_vector,user_id in user_data:
        user_dic[int(user_id)]=user_vector
    print(len(user_dic))
    return user_dic


def make_all_pro_paths(data_addr, data_class_addr, train_addr, test_addr ,validation_adr , eid_addr , user_addr):

    eids = {}
#    eids['train'] =[]
#    eids['test'] =[]
#    eids['validation']=[]
    with open(eid_addr) as file:
        eids = json.load(file)

    user_dic = make_user_dic(user_addr)

    data_dir = data_addr
    labels = {}
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = int(line[line.index('eid:') + 4: line.index('label')])
            label = int(line[line.index('label:') + 6])
            labels[eid] = label

    with open(train_addr, 'wt') as train_f, open(test_addr, 'wt') as test_f , open(validation_adr, 'wt') as valid_f:
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                eid = int(filename.split('.')[0])
                label = labels[eid]
                if label==1:
                    with open(data_addr + '/' + filename, encoding="utf8") as f:
                        event_data = json.load(f)
                        generated_cascades = make_cascades(event_data , user_dic)
                        for item in generated_cascades:

                            if eid in eids['train'] and random.random()<0.4 :
                                cascade = convert_to_seq(item[0])
                                train_f.write(cascade + '\n')
                            
                            elif random.random()<0.05 :
                                cascade = convert_to_seq(item[0])
                                if eid in eids['test']:
                                    test_f.writelines(cascade + '\n')
                                else:
                                    valid_f.writelines(cascade + '\n')


def convert_to_user_id(cascade):
    cascade= np.array([int(item.id) for item in cascade])
    cascade = cascade.tolist()
    return cascade

def conver_to_user_vector(cascade , user_dic):
    new_cascade =[]
    new_cascade.extend([user_dic[item.id] for item in cascade])
    new_cascade = np.array(new_cascade)
    new_cascade = new_cascade.tolist()
    return new_cascade

def convert_to_feature_vector(cascade , user_dic , d2v_model):
    new_cascade=[]
    for item in cascade:
        text = item.text.lower().split()
        text_embeding = d2v_model.infer_vector(text)
        user = user_dic[item.id]
        vector = np.concatenate((text_embeding,user) , axis=0)
        new_cascade.append(vector)
    new_cascade = np.array(new_cascade)
    new_cascade = new_cascade.tolist()
    return new_cascade

def convert_to_seq(cascade):
    seq_str =''
    for user in cascade:
        seq_str += str(user.id)+',' + str(user.time_of_tweet)+' '
    return seq_str





def make_cascades(event , user_dic):
    engaged_users = []
    for tweet in event:
        if User(tweet).id in user_dic:
            engaged_users.append(User(tweet))
    generated_cascades =[]
    for cas_len in cas_lens:
        if len(engaged_users) >  cas_len +1 :
            cascade = engaged_users[0:cas_len]
            end = cas_len+pred_len if cas_len+pred_len <= len(engaged_users) else len(engaged_users)
            next_cascade = engaged_users[cas_len:end]
            generated_cascades.append([cascade , next_cascade])
#            break


    return generated_cascades


def read_all_events(data_addr, data_class_addr, eids_addr):
    with open(eids_addr) as file:
        eids = json.load(file)

    data_dir = data_addr
    labels = {}
    all_events = {}
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = int(line[line.index('eid:') + 4: line.index('label')])
            label = int(line[line.index('label:') + 6])
            labels[eid] = label
    for file in os.listdir(data_dir):
        filename = file
        if filename.endswith(".json"):
            eid = int(filename.split('.')[0])
            if eid in eids['train']:
                with open(data_addr + '/' + filename, encoding="utf8") as f:
                    event_data = json.load(f)
                    all_events[eid] = {}
                    all_events[eid]['event'] = event_data
                    all_events[eid]['label'] = labels[eid]
    return all_events


def save_d2v_models(data_addr, data_class_addr):
    texts = []
    print('reading events..')
    all_events = read_all_events(data_addr, data_class_addr, '/media/external_3TB/3TB/rafie/paper/weibo_all/eids')
    print('reading texts..')

    for eid, e_data in all_events.items():
        event = e_data['event']
        for tweet in event:
            texts.append(tweet['text'])


    learn_model(texts, text_model_addr)


def learn_model(data, addr):
    model = Doc2Vec(vector_size=50,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    tagged_data = [TaggedDocument(_d.lower().split(), [i]) for i, _d in enumerate(data)]

    model.build_vocab(tagged_data)

    for epoch in range(10):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(addr)

    print("Model Saved in", addr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--validation', type=str, required=True)
    # parser.add_argument('--eids', type=str, required=True)
    args = parser.parse_args()


    # save_d2v_models('/media/external_3TB/3TB/rafie/rumdect/Weibo', '/media/external_3TB/3TB/rafie/rumdect/Weibo.txt')

    make_all_users('/media/external_3TB/3TB/rafie/rumdect/Weibo',
                  '/media/external_3TB/3TB/rafie/rumdect/Weibo.txt',
                  '/home/rafie/NeuralDiffusionModel-master/data/weibof/users_limited.txt')
    make_all_pro_paths('/media/external_3TB/3TB/rafie/rumdect/Weibo',
                       '/media/external_3TB/3TB/rafie/rumdect/Weibo.txt',
                       '/home/rafie/NeuralDiffusionModel-master/data/weibof/'+args.train ,
                       '/home/rafie/NeuralDiffusionModel-master/data/weibof/'+args.test ,
                       '/home/rafie/NeuralDiffusionModel-master/data/weibof/'+args.validation,
                       '/media/external_3TB/3TB/rafie/paper/weibo_all/eids',
                       '/home/rafie/NeuralDiffusionModel-master/data/weibof/users_limited.txt')

