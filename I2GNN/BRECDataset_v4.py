import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm
import pickle

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def load_graphml(path):
    data_list = []
    with open(path, "rb") as f:
        while True:
            try:
                graph = from_networkx(nx.convert_node_labels_to_integers(nx.parse_graphml(pickle.load(f))))
                if graph.x is not None:
                    graph.num_nodes = len(graph.x)
                    graph.x = graph.x.unsqueeze(1).to(dtype=torch.float32)
                data_list.append(graph)
            except EOFError:
                break
    return data_list


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split='original'
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[['original', 'CCoHG'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy", "brec_CCoHG.graphml"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt", "brec_CCoHG.pt"]

    def process(self):

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) if i < 320*64 or (i >= 380*64 and i<(400+320)*64) or (i>(400+380)*64)
                         else torch_geometric.data.Data(node_to_subgraph2=torch.empty((0)), z=torch.empty((0, 4)),
                                                        edge_index=torch.empty((2, 0)), center_idx=torch.empty((0, 2)),
                                                        subgraph2_to_subgraph=torch.empty((0)), num_nodes=0,
                                                        num_subgraphs=0, node_to_original_node=torch.empty((0)),
                                                        num_original_nodes=0, subgraph_to_graph=torch.empty((0)),
                                                        subgraph2_to_graph=torch.empty((0)), dtype=torch.int)
                         for i, data in enumerate(tqdm(data_list))]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        data_list = load_graphml(self.raw_paths[1])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[1])


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
