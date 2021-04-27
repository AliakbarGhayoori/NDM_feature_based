''' Translate input text with trained model. '''

import torch
import random
import argparse
from tqdm import tqdm
import numpy as np
from transformer.Translator_beam import Translator
import transformer.Constants as Constants
import pickle
from DataLoader import DataLoader
from sklearn.neighbors import NearestNeighbors


idx2vec_addr=  '/home/rafie/NeuralDiffusionModel-master/data/weibof/idx2vec.pickle'

CUDA = 1



def getF1(ground_truth, pred_cnt):
    right = np.dot(ground_truth, pred_cnt)
    pred = np.sum(pred_cnt)
    total = np.sum(ground_truth)
    print(right , pred , total)

    if pred == 0:
        return 0, 0, 0
    precision = right / pred
    recall = right / total
    if precision == 0 or recall == 0:
        return 0, 0, 0
    return (2 * precision * recall) / (precision + recall), precision, recall


def getMSE(ground_truth, pred_cnt):
    return mean_squared_error(ground_truth, pred_cnt)


def getMAP(ground_truth, pred_cnt):
    size_cascade = list(ground_truth).count(1)
    avgp = 0.
    ite_list = [[idx, item, ground_truth[idx]] for idx, item in enumerate(pred_cnt)]
    ite_list = sorted(ite_list, key=lambda x: x[1], reverse=True)
    n_positive = 0
    idx = 0
    while n_positive < size_cascade:
        if ite_list[idx][2] == 1:
            n_positive += 1
            pre = n_positive / (idx + 1.)
            avgp = avgp + pre
        idx += 1
        # if idx >= 1:
        #    break
    avgp = avgp / size_cascade
    return avgp


def ranking(seq , candidates , n ,  idx2vec):
    seq_user = np.array([idx2vec[int(idx.data.cpu().numpy())] for idx in seq[0]])
    candidates_user = [idx2vec[int(idx.data.cpu().numpy())] for idx in candidates[0]]
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(candidates_user)
    print('************',candidates_user, seq_user, np.mean(seq_user , axis =0))
    distances, indices = nbrs.kneighbors([np.mean(seq_user , axis =0)])
    output = [candidates[0][index] for index in indices]
    return output




def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    '''
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    '''
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=100,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    #    parser.add_argument('-seq_len', type=int ,default=20)
    parser.add_argument('-pred_len', type=int, default=15)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader

    test_data = DataLoader(use_valid=True, batch_size=opt.batch_size, cuda=opt.cuda)

    translator = Translator(opt)
    translator.model.eval()

    numuser = test_data.user_size

    num_right = 0
    num_total = 0

    avgF1 = 0
    avgPre = 0
    avgRec = 0
    
    avgF1_best = 0
    avgPre_best = 0
    avgRec_best = 0


    avgF1_long = 0
    avgPre_long = 0
    avgRec_long = 0

    avgF1_short = 0
    avgPre_short = 0
    avgRec_short = 0
    numseq = 0  # number of test seqs

    # for micro pre rec f1
    right = 0.
    pred = 0.
    total = 0.
    rigth_best =0.
    pred_best = 0.
    total_best =0.
    right_long = 0.
    pred_long = 0.
    total_long = 0.
    right_short = 0.
    pred_short = 0.
    total_short = 0.



    with open(idx2vec_addr, 'rb') as handle:
        idx2vec = pickle.load(handle)

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
            n_best = opt.pred_len
            opt.pred_len = batch.size(1) - 10
            opt.seq_len = 10
            all_samples = translator.translate_batch(batch[:, 0:opt.seq_len], opt.pred_len).data
            candidates = all_samples.view(1,-1)
#            best_candidates = ranking(batch.data[:, 0:opt.seq_len] , candidates , n_best , idx2vec)
            for bid in range(batch.size(0)):
                numseq += 1.0

                ground_truth = np.zeros([numuser])
                num_ground_truth = 0
                for user in batch.data[bid][1 + opt.seq_len:-1]:
                    if user == Constants.EOS or user == Constants.PAD:
                        break
                    ground_truth[user] = 1.0
                    num_ground_truth += 1
                pred_cnt = np.zeros([numuser])
                for beid in range(opt.beam_size):
                    for pred_uid in all_samples[bid, beid, 1 + opt.seq_len:num_ground_truth + 1]:
                        if pred_uid == Constants.EOS:
                            break
                        else:
                            pred_cnt[pred_uid] += 1.0 / opt.beam_size
#                pred_cnt_best = np.zeros([numuser])
#                for user in best_candidates:
#                    if user != Constants.EOS:
#                        pred_cnt_best[user] += 1.0

                F1, pre, rec = getF1(ground_truth, pred_cnt)
                avgF1 += F1
                avgPre += pre
                avgRec += rec
                right += np.dot(ground_truth, pred_cnt)
                pred += np.sum(pred_cnt)
                total += np.sum(ground_truth)
                
#                F1, pre, rec = getF1(ground_truth, pred_cnt_best)
#                avgF1_best += F1
#                avgPre_best += pre
#                avgRec_best += rec
#                right_best += np.dot(ground_truth, pred_cnt_best)
#                pred_best += np.sum(pred_cnt_best)
#                total_best += np.sum(ground_truth)

                print(avgF1, avgPre, avgRec)

    print('[Info] Finished.')
    print('Macro')
    print(avgF1 / numseq)
    print(avgPre / numseq)
    print(avgRec / numseq)
    print('Micro')
    pmi = right / pred
    rmi = right / total
    print(2 * pmi * rmi / (pmi + rmi))
    print(pmi)
    print(rmi)

#    print('[Info] Finished.')
#    print('Macro')
#    print(avgF1_best / numseq)
#    print(avgPre_best / numseq)
#    print(avgRec_best / numseq)
#    print('Micro')
#    pmi = right_best / pred_best
#    rmi = right_best / total_best
#    print(2 * pmi * rmi / (pmi + rmi))
#    print(pmi)
#    print(rmi)


if __name__ == "__main__":
    main()
