''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants
import logging
import pickle
import json

class Options(object):
    
    def __init__(self):
        #data options.

        #train file path.
        self.train_data = 'data/weibo2/cascade.txt'

        #test file path.
        self.test_data = 'data/weibo2/cascadetest.txt'

        self.u2idx_dict = 'data/weibo2/u2idx.pickle'

        self.idx2u_dict = 'data/weibo2/idx2u.pickle'
        
        self.idx2vec_dict = 'data/weibo2/idx2vec.pickle'
        
        self.user_data = 'data/weibo2/users_limited.txt'

        #save path.
        self.save_path = ''

        self.batch_size = 32

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, use_valid=False, load_dict=True, cuda=True, batch_size=32, shuffle=True, test=False):
        self.options = Options()
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self._idx2vec = {}
        self.use_valid = use_valid
        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2vec_dict, 'wb') as handle:
                pickle.dump(self._idx2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)

        self._train_cascades = self._readFromFile(self.options.train_data)
        self._test_cascades = self._readFromFile(self.options.test_data)
        self.train_size = len(self._train_cascades)
        self.test_size = len(self._test_cascades)
        print("user size:%d" % (self.user_size-2)) # minus pad and eos
        print("training set size:%d    testing set size:%d" % (self.train_size, self.test_size))

        self.cuda = cuda
        self.test = test
        if not self.use_valid:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random.shuffle(self._train_cascades)

    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        test_user_set = set()

        lineid=0
        for line in open(opts.train_data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(line)
                    print(chunk)
                    print(lineid)
                train_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        self._idx2vec[pos] = [pos]*8
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        self._idx2vec[pos] = [pos]*8
        pos += 1
        
        user_data = [json.loads(d) for d in open(opts.user_data, "rt").readlines()]
        user_dic = {}
        for user_vector, user_id in user_data:
            user_dic[user_id] = user_vector

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2vec[pos] = user_dic[int(user)]
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))


    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])
                    #if len(userlist) > 500:
                    #    break
                    # uncomment these lines if your GPU memory is not enough

            if len(userlist) > 1:
#                userlist.append(Constants.EOS)
                t_cascades.append(userlist)
        return t_cascades

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)
            lens = [len(inst) for inst in insts]


            src=[]
            trg_data=[]
            for i in range(len(insts)):
                seq_len = lens[i]
                src.append(insts[i][:int(seq_len/2)])
                trg_data.append([Constants.EOS]+insts[i][int(seq_len/2):seq_len])
            
            
        

            max_len_src = max(len(inst) for inst in src)
            src = np.array([inst + [Constants.PAD] * (max_len_src - len(inst))
                                    for inst in src])
            src = Variable(torch.LongTensor(src), volatile=self.test)
                                    
            if self.cuda:
                src = src.cuda(0)
                    
            max_len_trg = max(len(inst) for inst in trg_data)
            trg_data = np.array([inst + [Constants.PAD] * (max_len_trg - len(inst))
                                    for inst in trg_data])
            trg_data = Variable(torch.LongTensor(trg_data), volatile=self.test)
                                    
            if self.cuda:
                trg_data = trg_data.cuda(0)
            
            trg = trg_data[:,:-1]
            trg_y = trg_data[:,1:]
            
            
            src_mask = (src != Constants.PAD).unsqueeze(-2)
            trg_mask = (trg_y != Constants.PAD).unsqueeze(-2)
    
            src_lengths = [int(x/2) for x in lens]
            trg_lengths = [x/2+1 if x%2==0 else int(x/2)+1 for x in lens]
            
            src_lengths = np.array(src_lengths)
            trg_lengths = np.array(trg_lengths)
            
            reverse_idx_src = np.argsort(-src_lengths)
            reverse_idx_trg = np.argsort(-trg_lengths)
            
            src=src[reverse_idx_src]
            src_mask = src_mask[reverse_idx_src]
            src_lengths = src_lengths[reverse_idx_src]
            
            trg=trg[reverse_idx_trg]
            trg_mask = trg_mask[reverse_idx_trg]
            trg_lengths = trg_lengths[reverse_idx_trg]
            try_y = trg_y[reverse_idx_trg]



#            print(insts[10],src[10],src_mask[10],src_lengths[10],trg[10],trg_mask[10],trg_lengths[10])


            return src,src_mask,src_lengths,trg,trg_y,trg_mask,trg_lengths
                
        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if not self.use_valid:
                seq_insts = self._train_cascades[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
            src,src_mask,src_lengths,trg,trg_y,trg_mask,trg_lengths = pad_to_longest(seq_insts)
            #print('???')
            #print(seq_data.data)
            #print(seq_data.size())
            return src,src_mask,src_lengths,trg,trg_y,trg_mask,trg_lengths
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)
                #random.shuffle(self._test_cascades)

            self._iter_count = 0
            raise StopIteration()
