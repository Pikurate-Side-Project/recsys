from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class NCFCollator():
    
    def __call__(self, samples: List[Dict[str, int]]) -> Any:
        targets = [[int(s['user']), int(s['url'])] for s in samples]
        labels = [int(s['label']) for s in samples]
        
        return_value = {
            'targets': torch.tensor(targets, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
        
        return return_value


class NCFDataset(Dataset):

    def __init__(self, users: List[int], urls: List[int], labels: List[int]) -> None:
        self.users = users
        self.urls = urls
        self.labels = labels

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, item) -> Dict[str, int]:
        user = self.users[item]
        url = self.urls[item]
        label = self.labels[item]

        return {
            'user': user,
            'url': url,
            'label': label
        }