''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np


from transformer.MyModel import EncoderDecoder , Encoder , Decoder , Generator ,BahdanauAttention


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt
        
        attention = BahdanauAttention(model_opt.d_inner_hid)
        model= EncoderDecoder(  Encoder(model_opt.d_word_vec, model_opt.d_inner_hid, num_layers=1, dropout=model_opt.dropout),
                                Decoder(model_opt.d_word_vec, model_opt.d_inner_hid, attention, num_layers=1, dropout=model_opt.dropout),
                                nn.Embedding(model_opt.user_size, model_opt.d_word_vec),
                                nn.Embedding(model_opt.user_size, model_opt.d_word_vec),
                                Generator(model_opt.d_inner_hid, model_opt.user_size))
                                


        prob_projection = nn.Softmax()
        model_dict = checkpoint['model']
#        new_state_dict = OrderedDict()
#        for k, v in state_dict.items():
#            name = k[7:]
#            new_state_dict[name] = v

        model.load_state_dict(model_dict)
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda(0)
            prob_projection.cuda(0)
        else:
            print('no cuda')
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def translate_batch(self,src,src_mask,src_lengths,trg,trg_mask,trg_lengths):
        ''' Translation work in one batch '''
#        self.opt.beam_size=100

        # for i in range(35):
        #     with torch.no_grad():
        #         self.encoder_hidden, self.encoder_final = self.model.encode(src, src_mask, src_lengths)
        #         prev_y = torch.ones(1, 1).fill_(0).type_as(trg)
        #         trg_mask = torch.ones_like(prev_y)
        #
        #     output = []
        #     attention_scores = []
        #     hidden = None
        #     for j in range(self.opt.beam_size):
        #
        #


        output_f=[]
        for j in range(self.opt.beam_size):
            with torch.no_grad():
                encoder_hidden, encoder_final = self.model.encode(src, src_mask, src_lengths)
                prev_y = torch.ones(1, 1).fill_(0).type_as(trg)
                trg_mask = torch.ones_like(prev_y)
    
            output = []
            attention_scores = []
            hidden = None
    
            for i in range(35):
                with torch.no_grad():
                    out, hidden, pre_output = self.model.decode(encoder_hidden, encoder_final, src_mask,
                                                           prev_y, trg_mask, hidden)
#                    print(hidden,prev_y)

                    prob = self.model.generator(pre_output[:, -1])
#                    print(prob)
#                print(prob)
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data.item()
                output.append(next_word)
                prev_y = torch.ones(1, 1).type_as(trg).fill_(next_word)
                attention_scores.append(self.model.decoder.attention.alphas.cpu().numpy())
    
            output = np.array(output)
            output_f.append(output)
    
        output_f = np.array(output_f)
        output_f2=[]
        output_f2.append(output_f)
        output_f2 = np.array(output_f2)

        print(output_f2)
        return output_f2


    def predict_next_user(self,num,output,hidden,attention_scores,src_mask,prev_y):
        with torch.no_grad():
            out, hidden, pre_output = self.model.decode(encoder_hidden, encoder_final, src_mask,
                                                        prev_y, trg_mask, hidden)

            prob = self.model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(trg).fill_(next_word)
        attention_scores.append(self.model.decoder.attention.alphas.cpu().numpy())
        return next_word,prev_y,attention_scores

