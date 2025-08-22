import torch_geometric
from torch_geometric.data import InMemoryDataset
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
                                                 split="CCoHG"),
                      BRECDataset_v4.BRECDataset(name=name,
                                                 root=root,
                                                 transform=transform,
                                                 pre_transform=pre_transform,
                                                 pre_filter=pre_filter,
                                                 split="3r2r")
                      ]

    def __len__(self):
        return sum([len(brec) for brec in self.BRECs])

    def __getitem__(self, item):
        if type(item) is int:
            item = slice(item, item+1, 1)
        original = 25600
        CCoHG = 6400
        _3r2r = 6400

        value = original
        # Original typical BREC graphs
        if (type(item) is slice and item.start < original and item.stop-1 <original) or (type(item) is int and item < original):
            return self.BRECs[0][item]

        # Added in CCoHG graphs
        old = value
        value = value+CCoHG
        if (type(item) is slice and item.start < value and item.stop-1 < value) or (type(item) is int and item < value):
            return self.BRECs[1][slice(item.start-old, item.stop-old, item.step)]

        # Added in 3r2r graphs
        old = value
        value = value + _3r2r
        if (type(item) is slice and item.start < value and item.stop - 1 < value) or (
                type(item) is int and item < value):
            return self.BRECs[2][slice(item.start - old, item.stop - old, item.step)]

        # Original random BREC graphs at the end
        value = value + original
        if (type(item) is slice and item.start < value and item.stop - 1 < value) or (
                type(item) is int and item < value):
            return self.BRECs[0][slice(item.start-CCoHG-_3r2r,
                                       item.stop-CCoHG-_3r2r, item.step)]

        # CCoHG graphs at the end
        value = value + CCoHG
        if (type(item) is slice and item.start < value and item.stop - 1 < value) or (
                type(item) is int and item < value):
            return self.BRECs[1][slice(item.start-(2*original)-_3r2r,
                                       item.stop-(2*original)-_3r2r, item.step)]

        # 3r2r graphs at the end
        value = value + _3r2r
        if (type(item) is slice and item.start < value and item.stop - 1 < value) or (
                type(item) is int and item < value):
            return self.BRECs[2][slice(item.start - (2 * original) - (2*CCoHG),
                                       item.stop - (2 * original) - (2*CCoHG), item.step)]


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
