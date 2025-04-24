import torch
import torch.nn as nn
from torch.autograd import Variable

from . import rits_i  # forward and backward imputers

SEQ_LEN = 10  # adjust if needed


class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight):
        super(Model, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight

        self.build()

    def build(self):
        self.rits_f = rits_i.Model(133, self.rnn_hid_size, self.impute_weight)
        self.rits_b = rits_i.Model(133, self.rnn_hid_size, self.impute_weight)

    def forward(self, data):
        # Run forward and backward imputations

        ret_f = self.rits_f(data, direct='forward')
        ret_b = self.reverse(self.rits_b(data, direct='backward'))

        return self.merge_ret(ret_f, ret_b)

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        total_loss = loss_f + loss_b + loss_c
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        eval_masks = torch.logical_or(ret_f['eval_masks'], ret_b['eval_masks'])
        merged_evals = (ret_f['evals'] * ret_f['eval_masks'] + ret_b['evals'] * ret_b['eval_masks']) / (ret_f['eval_masks'] + ret_b['eval_masks'] + 1e-5)  # Weighted average


        return {
            'loss': total_loss,
            'imputations': imputations,
            'evals' : merged_evals,
            'eval_masks' : eval_masks
        }

    def get_consistency_loss(self, pred_f, pred_b):
        return torch.abs(pred_f - pred_b).mean() * 1e-1

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = torch.arange(tensor_.size(1) - 1, -1, -1).long().to(tensor_.device)
            return tensor_.index_select(1, indices)

        return {key: reverse_tensor(val) for key, val in ret.items()}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
