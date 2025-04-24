import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 10

class MySet(Dataset):
    def __init__(self, split='train'):
        super(MySet, self).__init__()
        self.content = open('./json/improve_brits_data.json').readlines()

        indices = np.arange(len(self.content))
        val_indices = set(np.random.choice(indices, len(self.content) // 5).tolist())
        
        if split == 'train':
            self.indices = [i for i in indices if i not in val_indices]
        elif split == 'val':
            self.indices = [i for i in val_indices]
        else:
            self.indices = indices
        print(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        rec = json.loads(self.content[real_idx])
        rec['is_train'] = 1 if real_idx not in self.indices else 0
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        return {
            'values': torch.FloatTensor([r['values'] for r in recs]),
            'forwards': torch.FloatTensor([r['forwards'] for r in recs]),
            'masks': torch.FloatTensor([r['masks'] for r in recs]),
            'deltas': torch.FloatTensor([r['deltas'] for r in recs]),
            'evals': torch.FloatTensor([r['evals'] for r in recs]),
            'eval_masks': torch.FloatTensor([r['eval_masks'] for r in recs]),
        }


    return {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward),
        'is_train': torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))
    }

def get_loader(batch_size=64, shuffle=True, split='train'):
    data_set = MySet(split=split)
    data_iter = DataLoader(dataset=data_set,
                           batch_size=batch_size,
                           num_workers=1,
                           shuffle=shuffle,
                           pin_memory=True,
                           collate_fn=collate_fn)
    return data_iter

