import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

SEQ_LEN = 50  # you can modify this depending on your data window size

class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.W = Parameter(torch.Tensor(rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        return torch.exp(-gamma)

class Model(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight

        self.rnn_cell = nn.LSTMCell(input_size * 2, rnn_hid_size)
        self.regression = nn.Linear(rnn_hid_size, input_size)
        self.temp_decay = TemporalDecay(input_size=input_size, rnn_hid_size=rnn_hid_size)

    def forward(self, data, direct='forward'):
        values = data[direct]['values']      # [B, T, D]
        masks = data[direct]['masks']        # [B, T, D]
        deltas = data[direct]['deltas']      # [B, T, D]
        evals = data[direct]['evals']        # [B, T, D]
        eval_masks = data[direct]['eval_masks']  # [B, T, D]
        
        B, T, D = values.size()
        h = torch.zeros(B, self.rnn_hid_size, device=values.device)
        c = torch.zeros(B, self.rnn_hid_size, device=values.device)

        imputations = []
        x_loss = 0.0

        # Process data in chunks of SEQ_LEN
        for start in range(0, T, SEQ_LEN):
            end = min(start + SEQ_LEN, T)
            seq_values = values[:, start:end, :]  # [B, SEQ_LEN, D]
            seq_masks = masks[:, start:end, :]      # [B, SEQ_LEN, D]
            seq_deltas = deltas[:, start:end, :]    # [B, SEQ_LEN, D]
            seq_evals = evals[:, start:end, :]      # [B, SEQ_LEN, D]
            seq_eval_masks = eval_masks[:, start:end, :]  # [B, SEQ_LEN, D]

            for t in range(seq_values.size(1)):
                x = seq_values[:, t, :]
                m = seq_masks[:, t, :]
                d = seq_deltas[:, t, :]

                gamma = self.temp_decay(d)
                h = h * gamma

                x_h = self.regression(h)
                x_c = m * x + (1 - m) * x_h  # composite input

                # Compute imputation loss vs. ground truth evals
                target = seq_evals[:, t, :]
                target_mask = seq_eval_masks[:, t, :]
                x_loss += torch.sum(torch.abs(x_h - target) * target_mask) / (torch.sum(target_mask) + 1e-5)

                inputs = torch.cat([x_c, m], dim=1)
                h, c = self.rnn_cell(inputs, (h, c))

                imputations.append(x_h.unsqueeze(1))

        imputations = torch.cat(imputations, dim=1)  # [B, T, D]

        return {
            'loss': x_loss * self.impute_weight,
            'imputations': imputations,
            'evals': evals,
            'eval_masks': eval_masks
        }

    def run_on_batch(self, data, optimizer=None, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
