import os
import json
import random
from multiprocessing import Pool
from io import open
from pathlib import Path
from pathlib import Path
import calendar
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
from sklearn import preprocessing


import numpy as npe
from tqdm import tqdm
import argparse
import numpy as np


from utils import TUser
import random

global all_event
global less_than_limit_event
all_event = 0
less_than_limit_event = 0
users = {}
db_config = {}
global limit
train_ratio = 0.8
t = 60
f_model_addr = 'd2v/fd2v_t.model'
r_model_addr = 'd2v/rd2v_t.model'
cas_lens = [50,40,30,20,10]



def make_all_users(data_addr,data_class_addr, users_addr):
    data_dir = data_addr
    engaged_users={}
    user_id ={}
    labels = {}
    seqs={}
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = str(line[line.index('eid:') + 4: line.index('label')-1])
            label = int(line[line.index('label:') + 6])
            labels[eid] = label
            seq = line[line.index('label:') + 8 : line.index('\n')]
            seq = seq.split()
            seqs[eid] = seq

    for eid, label in labels.items():
        tweet_ind = 0
        while ( tweet_ind < len(seqs[eid])):
                tweet = seqs[eid][tweet_ind]
                tweet_ind += 1
                filename = tweet + '.txt'
                addr = data_addr + '/' + eid + '-' + str(label)
            
                if(Path(addr + '/' + filename).exists()):
                    with open(addr + '/' + filename, encoding="utf8") as f:
                        event_data = json.load(f)
                        user = event_data['tweet']['user']
                        user = TUser(user)
                        if user.id in engaged_users:
                            engaged_users[user.id]+=1
                        else:
                            engaged_users[user.id]=1
                            user_id[user.id] = user


    print('before all users')
    all_users = list(set([user_id[userid] for userid,count in engaged_users.items() if count>0]))
    
    print('after all usesrs')
    user_embedding =[]
    for user in all_users:
        user_embedding.append(user.vector[0:6])
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(user_embedding)
    user_embedding = scaler.transform(user_embedding)
    print('before embedding')
    
    with open(users_addr, 'wt') as users_f:
        for i, user in enumerate(all_users):
            new_user_vector = user.vector.tolist()
            # print('before' , new_user_vector)
            new_user_vector[0:6] = user_embedding[i]
            # print('after' , new_user_vector ,user_embedding[i])
            user_id = user.id
            json_data = json.dumps([new_user_vector , int(user_id)])
            users_f.write(json_data + '\n')




def make_all_pro_paths(data_addr, data_class_addr, train_addr='', test_addr='' ,validation_adr='' ,eid_addr=''):
    
    empty_eids = 0
    all_tweets = 0
    empty_tweets = 0
    fake_events = 0
    real_events = 0
    all_len=[]
    all_time =[]
    
#    with open(eid_addr) as file:
#        eids = json.load(file)

    labels = {}
    seqs = {}
    #    r_model = Doc2Vec.load(r_model_addr)
    #    f_model = Doc2Vec.load(f_model_addr)
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = str(line[line.index('eid:') + 4: line.index('label')-1])
            label = int(line[line.index('label:') + 6])
            seq = line[line.index('label:') + 8 : line.index('\n')]
            seq = seq.split()
            seqs[eid] = seq
            labels[eid] = label
            all_tweets += len(seq)
    
    with open(train_addr, 'wt') as train_f, open(test_addr, 'wt') as test_f, open(validation_adr, 'wt') as valid_f:
        for eid, label in labels.items():
            
            engaged_users = []
            tweet_ind = 0
            while ( tweet_ind < len(seqs[eid])):
                tweet = seqs[eid][tweet_ind]
                tweet_ind += 1
                filename = tweet + '.txt'
                addr = data_addr + '/' + eid + '-' + str(label)
                
                if(Path(addr + '/' + filename).exists()):
                    with open(addr + '/' + filename, encoding="utf8") as f:
                        event_data = json.load(f)
                        user = event_data['tweet']['user']
                        engaged_users.append(user)
        
            if label==1 or label==0:
                for cas_len in cas_lens:
                    if len(engaged_users) >  cas_len +1 :
                        cascade = [TUser(user).id for user in engaged_users]
                        cascade = cascade[0:cas_len]
                        cascade = convert_to_seq(cascade)
                        
                        if random.random()<0.8:
                            train_f.write(cascade + '\n')
                        else:
                            if random.random()<0.5:
                                test_f.writelines(cascade + '\n')
                            else:
                                valid_f.writelines(cascade + '\n')


def convert_to_seq(cascade):
    seq_str =''
    for user in cascade:
        seq_str += str(user)+',' + str(0)+' '
    return seq_str


def make_propagation_path(event):
    global all_event
    global less_than_limit_event
    global limit
    all_event+=1
    engaged_users = []
    for tweet in event:
        engaged_users.append(tweet)
    if len(engaged_users) < limit:
        engaged_users.extend([random.choice(engaged_users) for i in range(limit - len(engaged_users))])
        less_than_limit_event+=1
    engaged_users = engaged_users[0:limit]
    
    engaged_users_vector = [User(user).vector for user in engaged_users]
    return engaged_users_vector


def read_all_events(data_addr, data_class_addr,eids_addr):
    data_dir = data_addr
    labels = {}
    all_events = {}
    seqs = {}
    with open(eids_addr) as file:
        eids = json.load(file)
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = str(line[line.index('eid:') + 4: line.index('label') - 1])
            label = int(line[line.index('label:') + 6])
            seq = line[line.index('label:') + 8: line.index('\n')]
            seq = seq.split()
            seqs[eid] = seq
            labels[eid] = label
    
    for eid, label in labels.items():
        tweet_ind = 0
        #        if eid in eids['train']:
        while (tweet_ind < len(seqs[eid])):
            tweet = seqs[eid][tweet_ind]
            tweet_ind += 1
            filename = tweet + '.txt'
            addr = data_addr + '/' + eid + '-' + str(label)
            
            if (Path(addr + '/' + filename).exists()):
                with open(addr + '/' + filename, encoding="utf8") as f:
                    event_data = json.load(f)
                    all_events[tweet] = {}
                    all_events[tweet]['tweet'] = event_data
                    all_events[tweet]['label'] = label
    return all_events


def save_w2v_models(data_addr, data_class_addr):
    real_texts = []
    fake_texts = []
    print('reading events..')
    all_events = read_all_events(data_addr, data_class_addr, '/media/external_3TB/3TB/rafie/paper/twitter_all/eids')
    print('reading texts..')
    
    for tweet_id, tweet in all_events.items():
        label = tweet['label']
        t = tweet['tweet']['tweet']
        if label == 0:
            real_texts.append(t['text'])
        else:
            fake_texts.append(t['text'])
    
    learn_model(real_texts , r_model_addr)
    learn_model(fake_texts , f_model_addr)



def learn_model(data ,addr):
    model = Doc2Vec(size=50,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    tagged_data = [TaggedDocument(_d.lower().split(),[i]) for i, _d in enumerate(data)]
                    
    model.build_vocab(tagged_data)
                    
    for epoch in range(10):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                                    total_examples=model.corpus_count,
                                    epochs=model.iter)
                                    # decrease the learning rate
        model.alpha -= 0.0002
                                    # fix the learning rate, no decay
        model.min_alpha = model.alpha
                    
    model.save(addr)
    print("Model Saved in" , addr)


if __name__ == '__main__':
    global limit
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--train', type=str, required=True)
#    parser.add_argument('--test', type=str, required=True)
#    parser.add_argument('--validation', type=str, required=True)
#    args = parser.parse_args()

    
    
    # save_w2v_models('/media/external_3TB/3TB/rafie/results-retweet', '/media/external_3TB/3TB/rafie/rumdect/Twitter.txt')
    
    make_all_users('/media/external_3TB/3TB/rafie/politifact-raw-data/Politifact',
                   '/media/external_3TB/3TB/rafie/politifact-raw-data/Politifact.txt',
                   '/home/rafie/NeuralDiffusionModel-master/data/politifact/users_limited.txt')
    
#    make_all_pro_paths('/media/external_3TB/3TB/rafie/gossipcop-raw-data/Gossipcop', '/media/external_3TB/3TB/rafie/gossipcop-raw-data/Gossipcop.txt','/home/rafie/NeuralDiffusionModel-master/data/gossipcop/'+args.train , '/home/rafie/NeuralDiffusionModel-master/data/gossipcop/'+args.test , '/home/rafie/NeuralDiffusionModel-master/data/gossipcop/'+args.validation,'/media/external_3TB/3TB/rafie/paper/twitter_all/eids')
    print('done')
