import torch
import torch.nn as nn
from torch.autograd import Variable

from . import rits_i  # your forward RITS model

SEQ_LEN = 36

class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight

        self.build()

    def build(self):
        self.rits_f = rits_i.Model(self.rnn_hid_size, self.impute_weight)
        self.rits_b = rits_i.Model(self.rnn_hid_size, self.impute_weight)

    def forward(self, data):
        ret_f = self.rits_f(data, mode='forward')
        ret_b = self.reverse(self.rits_b(data, mode='backward'))

        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        # Compute total loss = forward + backward + consistency
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        total_loss = loss_f + loss_b + loss_c

        # Average imputations
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        # Final return dictionary
        return {
            'loss': total_loss,
            'imputations': imputations
        }

    def get_consistency_loss(self, pred_f, pred_b):
        return torch.abs(pred_f - pred_b).mean() * 1e-1

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = torch.arange(tensor_.size(1) - 1, -1, -1).long().to(tensor_.device)
            return tensor_.index_select(1, indices)

        reversed_ret = {k: reverse_tensor(v) for k, v in ret.items()}
        return reversed_ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
