import networkx as nx
import numpy as np
import random
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx, to_networkx
import pickle

torch_geometric.seed_everything(2022)


part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "CCoHG": (400, 500),
}
PAIR_NUM = 500
NUM_RELABEL = 32
# REPEAT_TEST_NUM = 10


def relabel(g6):
    pyg_graph = from_networkx(nx.from_graph6_bytes(g6))
    n = pyg_graph.num_nodes
    edge_index_relabel = pyg_graph.edge_index.clone().detach()
    index_mapping = dict(zip(list(range(n)), np.random.permutation(n)))
    for i in range(edge_index_relabel.shape[0]):
        for j in range(edge_index_relabel.shape[1]):
            edge_index_relabel[i, j] = index_mapping[edge_index_relabel[i, j].item()]
    edge_index_relabel = edge_index_relabel[
        :, torch.randperm(edge_index_relabel.shape[1])
    ]
    pyg_graph_relabel = torch_geometric.data.Data(
        edge_index=edge_index_relabel, num_nodes=n
    )
    g6_relabel = nx.to_graph6_bytes(
        to_networkx(pyg_graph_relabel, to_undirected=True), header=False
    ).strip()
    return g6_relabel


def generate_relabel(g6, num=0):
    if num == 0:
        num = NUM_RELABEL
    g6_list = [g6]
    g6_set = set(g6_list)
    for id in range(num - 1):
        g6_relabel = relabel(g6)
        g6_set.add(g6_relabel)
        while len(g6_set) == len(g6_list):
            g6_relabel = relabel(g6)
            g6_set.add(g6_relabel)
        g6_list.append(g6_relabel)
    return g6_list


def generate_relabel_v2(nx_graph, num=0):
    n = nx_graph.order()
    if num == 0:
        num = 32
    nx_graph_list = [nx_graph]
    while len(nx_graph_list) < num:
        nx_graph = nx.relabel_nodes(nx_graph, dict(zip(list(range(n)), np.random.permutation(n))))
        # This might not work as intended as equality is not checked as per identity
        if not True in [nx.utils.graphs_equal(graph, nx_graph) for graph in nx_graph_list]:
            nx_graph_list.append(nx_graph)
    return nx_graph_list


def reindex_to_interlacing(g6_relabel_tuple):
    return_list = []
    for (x, y) in zip(g6_relabel_tuple[0], g6_relabel_tuple[1]):
        return_list.extend([x, y])
    return return_list


def save_graphml(path, graphs):
    with open(path, "wb") as f:
        for graph in graphs:
            gml = nx.generate_graphml(graph)
            pickle.dump(chr(10).join(gml), f)


def load_graphml(path):
    data_list = []
    with open(path, "rb") as f:
        while True:
            try:
                data_list.append(nx.convert_node_labels_to_integers(nx.parse_graphml(pickle.load(f))))
            except EOFError:
                break
    return data_list


def work_relabel(data_list):
    out_list = []
    for i in range(0, len(data_list), 2):
        nx_tuple = (data_list[i], data_list[i + 1])
        nx_relabel = (generate_relabel_v2(nx_tuple[0]), generate_relabel_v2(nx_tuple[1]))
        out_list.extend(reindex_to_interlacing(nx_relabel))
    return out_list


def work_relabel_reliability(data_list):
    out_list = []
    for i in range(0, len(data_list), 2):
        flag = random.randint(0, 1)
        nx_tuple = (data_list[i], data_list[i + 1])
        nx_relabel = generate_relabel_v2(nx_tuple[flag], num=NUM_RELABEL * 2)
        out_list.extend(nx_relabel)
    return out_list


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [
            "basic.npy",
            "regular.npy",
            "str.npy",
            "extension.npy",
            "cfi.npy",
            "4vtx.npy",
            "dr.npy",
            "CCoHG_BREC.graphml"
        ]

    @property
    def processed_file_names(self):
        return ["brec_v3.npy", "brec_CCoHG.graphml"]

    def process(self):
        g6_list = []

        print(len(g6_list))

        # Basic graphs: 0 - 60
        basic = np.load(self.raw_paths[0])
        for i in range(0, basic.size, 2):
            g6_tuple = (basic[i].encode(), basic[i + 1].encode())
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # Simple regular graphs: 60 - 110
        regular = np.load(self.raw_paths[1])
        for i in range(0, regular.size // 2):
            g6_tuple = regular[i]
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # Strongly regular graphs: 110 - 160
        strongly_regular = np.load(self.raw_paths[2])
        for i in range(0, strongly_regular.size, 2):
            g6_tuple = (strongly_regular[i].encode(), strongly_regular[i + 1].encode())
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # Extension graphs: 160 - 260
        extension = np.load(self.raw_paths[3])
        for i in range(0, extension.size // 2):
            g6_tuple = (extension[i][0].encode(), extension[i][1].encode())
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # CFI graphs: 260 - 360
        cfi = np.load(self.raw_paths[4])
        for i in range(0, cfi.size // 2):
            g6_tuple = cfi[i]
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # 4-vertex condition graphs: 360 - 380
        vtx_4 = np.load(self.raw_paths[5])
        for i in range(0, vtx_4.size, 2):
            g6_tuple = (vtx_4[i].encode(), vtx_4[i + 1].encode())
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # Distance regular graphs: 380 - 400
        distance_regular = np.load(self.raw_paths[6])
        for i in range(0, distance_regular.size, 2):
            g6_tuple = (distance_regular[i], distance_regular[i + 1])
            g6_relabel_tuple = (generate_relabel(g6_tuple[0]), generate_relabel(g6_tuple[1]))
            g6_list.extend(reindex_to_interlacing(g6_relabel_tuple))

        print(len(g6_list))

        # basic graphs: 0 - 60
        for i in range(0, basic.size, 2):
            flag = random.randint(0, 1)
            g6_tuple = (basic[i].encode(), basic[i + 1].encode())
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)

        print(len(g6_list))
        
        
        # Simple regular graphs: 60 - 110
        for i in range(0, regular.size // 2):
            flag = random.randint(0, 1)
            g6_tuple = regular[i]
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)

        print(len(g6_list))
        

        # Strongly regular graphs: 110 - 160
        for i in range(0, strongly_regular.size, 2):
            flag = random.randint(0, 1)
            g6_tuple = (strongly_regular[i].encode(), strongly_regular[i + 1].encode())
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)

        print(len(g6_list))
        
        # Extension graphs: 160 - 260
        for i in range(0, extension.size // 2):
            flag = random.randint(0, 1)
            g6_tuple = (extension[i][0].encode(), extension[i][1].encode())
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)
        print(len(g6_list))

        # CFI graphs: 260 - 360
        for i in range(0, cfi.size // 2):
            flag = random.randint(0, 1)
            g6_tuple = cfi[i]
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)
        print(len(g6_list))

        # 4-vertex condition graphs: 360 - 380
        for i in range(0, vtx_4.size, 2):
            flag = random.randint(0, 1)
            g6_tuple = (vtx_4[i].encode(), vtx_4[i + 1].encode())
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)
        print(len(g6_list))

        # Distance regular graphs: 380 - 400
        for i in range(0, distance_regular.size, 2):
            flag = random.randint(0, 1)
            g6_tuple = (distance_regular[i], distance_regular[i + 1])
            g6_relabel = generate_relabel(g6_tuple[flag], num=NUM_RELABEL * 2)
            g6_list.extend(g6_relabel)

        print(len(g6_list))
        #np.save(self.processed_paths[0], g6_list)

        nx_list = []
        print(len(nx_list))

        # CCoHG graphs: 400 - 500
        CCoHG = load_graphml(self.raw_paths[7])
        nx_list.extend(work_relabel(CCoHG))

        print(len(nx_list))

        nx_list.extend(work_relabel_reliability(CCoHG))

        print(len(nx_list))

        save_graphml(self.processed_paths[1], nx_list)


def main():
    dataset = BRECDataset()


if __name__ == "__main__":
    main()
