import torch
import numpy as np
from itertools import product


class CategoriesSampler:
    def __init__(self, label, n_batches, ways, n_images, permute=True):
        self.n_batch = n_batches
        self.n_cls = ways
        self.n_per = n_images
        self.permute = permute

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        if not permute:
            self.batches = []
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                for c in classes:
                    l = self.m_ind[c]
                    pos = torch.randperm(len(l))[:self.n_per]
                    batch.append(l[pos])
                batch = torch.stack(batch).t().reshape(-1)
                self.batches.append(batch)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        if self.permute:
            for _ in range(self.n_batch):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                for c in classes:
                    l = self.m_ind[c]
                    pos = torch.randperm(len(l))[:self.n_per]
                    batch.append(l[pos])
                batch = torch.stack(batch).t().reshape(-1)
                yield batch
        else:
            for batch in self.batches:
                yield batch


class CategoriesSamplerMult:
    def __init__(self, label, n_batches, ways, n_images, *, n_combinations=1):
        self.n_batch = n_batches
        self.n_cls = ways
        self.n_per = n_images
        self.n_combinations = n_combinations

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            all_classes = list(product(range(len(self.m_ind)),
                                       range(len(self.m_ind))))
            classes = [all_classes[cl] for cl in torch.randperm(len(all_classes))[:self.n_cls]]
            for c1, c2 in classes:
                c1 = self.m_ind[c1]
                c2 = self.m_ind[c2]
                pos1 = torch.randperm(len(c1))[:self.n_per]
                assert len(pos1) == self.n_per
                pos2 = torch.randperm(len(c2))[:self.n_per]
                assert len(pos2) == self.n_per
                batch.append(c1[pos1])
                batch.append(c2[pos2])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
