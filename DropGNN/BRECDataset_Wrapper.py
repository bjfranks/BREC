import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
import os
from tqdm import tqdm
import BRECDataset_v4

torch_geometric.seed_everything(2022)

# This construction is due to the added graphs having varying amounts of node features. Originals have 0, CCoHG has 1,
# Peptides has multiple.
class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.BRECs = [BRECDataset_v4.BRECDataset(name=name,
                                                 root=root,
                                                 transform=transform,
                                                 pre_transform=pre_transform,
                                                 pre_filter=pre_filter,
                                                 split="original"),
                      BRECDataset_v4.BRECDataset(name=name,
                                                 root=root,
                                                 transform=transform,
                                                 pre_transform=pre_transform,
                                                 pre_filter=pre_filter,
                                                 split="CCoHG")
                      ]

    def __len__(self):
        return sum([len(brec) for brec in self.BRECs])

    def __getitem__(self, item):
        #Original typical BREC graphs
        value = 25600
        if (type(item) is slice and item.start < value and item.stop-1 <value) or (type(item) is int and item < value):
            return self.BRECs[0][item]
        #Added in CCoHG graphs
        else:
            value = 25600+6400
            if (type(item) is slice and item.start < value and item.stop-1 < value) or (type(item) is int and item < value):
                return self.BRECs[1][slice(item.start-25600, item.stop-25600, item.step)]
            #Original random BREC graphs at the end
            else:
                value = 25600 + 25600 + 6400
                if (type(item) is slice and item.start < value and item.stop - 1 < value) or (
                        type(item) is int and item < value):
                    return self.BRECs[0][slice(item.start-6400, item.stop-6400, item.step)]
                else:
                    return self.BRECs[1][slice(item.start - (2*25600), item.stop - (2*25600), item.step)]


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
