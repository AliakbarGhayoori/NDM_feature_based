''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.Models import Decoder

CUDA =0

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Decoder(
            model_opt.user_size,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            kernel_size=model_opt.window_size,
            finit=model_opt.finit,
            d_inner_hid=model_opt.d_inner_hid,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        prob_projection = nn.Softmax()

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        h0 = torch.zeros(1, opt.beam_size, 4 * model_opt.d_word_vec)

        if opt.cuda:
            model.cuda(CUDA)
            prob_projection.cuda(CUDA)
            h0 = h0.cuda(CUDA)
        else:
            print('no cuda')
            model.cpu()
            prob_projection.cpu()
            h0 = h0.cpu()

        model.prob_projection = prob_projection

        self.h0 = h0
        self.model = model
        self.model.eval()

    def translate_batch(self, batch, pred_len):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        tgt_seq = batch
        # print(batch.size())
        #        print(tgt_seq.shape)

        batch_size = tgt_seq.size(0)
        max_len = min(tgt_seq.size(1), 100)
        max_len2 = max_len + pred_len
        beam_size = self.opt.beam_size

        # - Decode
        # print(tgt_seq.data[0,:])

        dec_partial_seq = torch.LongTensor(
            [[[tgt_seq.data[j, k] for k in range(max_len)] for i in range(beam_size)] for j in range(batch_size)])
        # size: (batch * beam) x seq

        # wrap into a Variable
        dec_partial_seq = Variable(dec_partial_seq, volatile=True)
        if self.opt.cuda:
            dec_partial_seq = dec_partial_seq.cuda(CUDA)
        for i in range(pred_len + 1):
#            print('here')
            len_dec_seq = max_len + i
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            #            print(dec_partial_seq.shape)
            # -- Decoding -- #
            dec_output, *_ = self.model(dec_partial_seq, self.h0, generate=True)
#            print(dec_output.shape)
            dec_output = dec_output.view(dec_partial_seq.size(0), -1, self.model_opt.user_size)
#            print(dec_output.shape)
            dec_output = dec_output[:, -1, :]  # (batch * beam) * user_size
#            print(dec_output.shape)

            out = self.model.prob_projection(dec_output)
#            print(out.shape)
            sample = torch.multinomial(out, 1, replacement=True)
#            print(sample.shape)

            # batch x beam x 1
            # sample = sample.view(batch_size, beam_size, 1).contiguous()
            sample = sample.long()
            dec_partial_seq = torch.cat([dec_partial_seq, sample], dim=1)
            dec_partial_seq = dec_partial_seq.view(batch_size, beam_size, -1)
            # print(dec_partial_seq.size())

        # - Return useful information
#        print(dec_partial_seq.shape)
        return dec_partial_seq
