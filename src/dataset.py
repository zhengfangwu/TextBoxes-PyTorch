import torch
import torch.utils.data

class ICDARDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(ICDARDataset, self).__init__()

    
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return 0