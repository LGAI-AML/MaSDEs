import os
import torch
import torch.utils.data as data
import numpy as np


def data_loader(args):
    if args.data_set == "air_quality":
        train_dataset = air_quality(train=True)
        test_dataset = air_quality(train=False)
        
    train_loader = data.DataLoader(dataset = train_dataset,
                                   batch_size = args.B,
                                   num_workers=0,
                                   drop_last=True,
                                   shuffle=True)
        
    test_loader = data.DataLoader(dataset = test_dataset,
                                   batch_size = args.B,
                                   num_workers=0,
                                   drop_last=True,
                                   shuffle=False)    
    
    return train_loader, test_loader

class air_quality(data.Dataset):
    def __init__(
        self,
        train=True,
    ):
        super(air_quality, self).__init__()
        datasets = np.load("datasets/..") 
        self.train = train
        self.train_data = torch.FloatTensor(datasets["train_data"])
        self.train_masks = torch.FloatTensor(datasets["train_mask"])

        self.test_data = torch.FloatTensor(datasets["test_data"])
        self.test_masks = torch.FloatTensor(datasets["test_mask"])
    def __getitem__(self, idx):
        if self.train:
            data, mask = self.train_data[idx], self.train_masks[idx]
        else:
            data, mask = self.test_data[idx], self.test_masks[idx]
        return data, mask
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
