# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
#from BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch.nn import CosineEmbeddingLoss


import os.path as osp
from shutil import copy, rmtree
import pdb
import argparse
import random
import torch.nn.functional as F
import torch_geometric.transforms as T
import data_processing as dp
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs

# from torchmetrics import AUROC
# from k_gnn import TwoMalkin, ConnectedThreeMalkin, TwoLocal, ThreeMalkin, ThreeLocal

from utils import create_subgraphs, create_subgraphs2, check_graphlet
from count_models import I2GNN

from Xent_Loss import nt_bxent_loss
import pickle
from BRECDataset_Wrapper import BRECDataset


# torch_geometric.seed_everything(2022)
NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 600
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 120.12
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 320),
    # "4-Vertex_Condition": (360, 380),
    # "Distance_Regular": (380, 400),
    "Distance_Regular": (380, 400),
    "CCoHG": (400, 500),
    #"3r2r": (500, 600),
}
parser = argparse.ArgumentParser(description="I2GNN for counting experiments.")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
parser.add_argument("--SEED", type=int, default=SEED)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--device", type=int, default=0)

# General settings.
parser.add_argument(
    "--target", default=0, type=int
)  # 0 for detection of tri-cycle, 3,4,...,8 for counting of cycles
parser.add_argument("--ab", action="store_true", default=False)

# Base GNN settings.
parser.add_argument("--model", type=str, default="GNN")
parser.add_argument("--layers", type=int, default=4)

# Nested GNN settings
parser.add_argument(
    "--h",
    type=int,
    default=None,
    help="hop of enclosing subgraph;\
                    if None, will not use NestedGNN",
)
parser.add_argument("--max_nodes_per_hop", type=int, default=None)
parser.add_argument(
    "--node_label",
    type=str,
    default="hop",
    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".',
)
parser.add_argument(
    "--use_rd",
    action="store_true",
    default=False,
    help="use resistance distance as additional node labels",
)

# Training settings.
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay_factor", type=float, default=0.9)
parser.add_argument("--patience", type=int, default=10)

# Other settings.
parser.add_argument("--seed", type=int, default=233)
parser.add_argument(
    "--save_appendix",
    default="",
    help="what to append to save-names when saving results",
)
parser.add_argument(
    "--keep_old",
    action="store_true",
    default=False,
    help="if True, do not overwrite old .py files in the result folder",
)
parser.add_argument("--dataset", default="count_cycle")
parser.add_argument("--load_model", default=None)
parser.add_argument("--eval", default=0, type=int)
parser.add_argument("--train_only", default=0, type=int)

parser.add_argument(
    "--loss",
    type=str,
    default="CosineEmbeddingLoss",
    help="Options are ['CosineEmbeddingLoss', 'nt_bxent_loss']",#TODO
)
parser.add_argument("--loss_parameter", type=float, default=1)
parser.add_argument(
    '--parts',
    nargs='+',
    default=[],#list(part_dict.keys()),
    help='Options are a subset of '+str(part_dict.keys())
)
parser.add_argument("--name_tag", type=str, default=None)
parser.add_argument("--logging", type=str, default="default.log")
parser.add_argument("--root", type=str, default=".")

args = parser.parse_args()

P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
torch_geometric.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(name, device):
    time_start = time.process_time()

    # Do something
    def pre_transform2(g):
        return create_subgraphs2(
            g,
            args.h,
            max_nodes_per_hop=args.max_nodes_per_hop,
            node_label=args.node_label,
            use_rd=args.use_rd,
        )

    dataset = BRECDataset(name=name, pre_transform=pre_transform2)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device):
    time_start = time.process_time()

    # Do something
    kwargs = {
        "num_layers": args.layers,
        "edge_attr_dim": 1,
        # "target": args.target,
        # "y_ndim": 2,
    }

    dataset_none = None
    model = I2GNN(dataset_none, **kwargs).to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, device, args):
    """
    When testing on BREC, even on the same graph, the output embedding may be different,
    because numerical precision problem occur on large graphs, and even the same graph is permuted.
    However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
    some modification is needed to avoid computing the inverse matrix of a zero matrix.
    """
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                #torch.set_printoptions(threshold=10_000)
                #print(data.x)
                pred = model(data.to(device)).detach()
                #print(model.state_dict())
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
                #print(pred_0_list, pred_1_list)
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            #big = torch.max(torch.max(torch.abs(X)), torch.max(torch.abs(Y)))
            #X/=big
            #Y/=big
            D = X - Y
            #nanprint(D)
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = torch.where(torch.abs(D) < torch.maximum(torch.abs(X), torch.abs(Y))/100, 0, D) # Avoids false positives
            if log_flag:
                logger.info(f"newD = {D}")
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            # print(D, D_mean, S, inv_S)
            result = NUM_RELABEL*torch.mm(torch.mm(D_mean.T, inv_S), D_mean)
            if log_flag:
                logger.info(f"result = {result}")
            return result

    time_start = time.process_time()

    # Do something
    #num_nodes_list = np.load('num_node.npy', allow_pickle=True)
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)
    store = []

    ids = [] #[89, 112, 113, 114, 115, 116, 118, 119, 120, 121, 123, 124, 128, 129, 131, 134, 138,
           #139, 140, 142, 143, 149, 151, 152, 153, 154, 157, 158, 64, 111, 125, 127, 132, 133,
           #136, 137, 141, 144, 146, 150, 159, 117, 145, 148, 155, 156, 126, 130, 135, 147, 122]

    file = f"default.pkl"
    if args.name_tag is not None:
        file = f"{args.root}/{args.name_tag}.pkl"

    for part_name in args.parts: #for part_name, part_range in part_dict.items():
        part_range = part_dict[part_name]
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            time_start = time.process_time()
            epochs = 0
            for test_count in range(10):
                dataset_traintest = dataset[
                    id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
                ]
                model = get_model(args, dataset_traintest[0].num_nodes, 1, device)  # num_nodes_list[id]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer)  # StepLR(optimizer, gamma=0.5, step_size=250)
                dataset_reliability = dataset[
                    (id + SAMPLE_NUM)
                    * NUM_RELABEL
                    * 2 : (id + SAMPLE_NUM + 1)
                    * NUM_RELABEL
                    * 2
                ]
                model.train()
                for _ in range(EPOCH):
                    traintest_loader = torch_geometric.loader.DataLoader(
                        dataset_traintest, batch_size=BATCH_SIZE
                    )
                    loss_all = 0
                    for data in traintest_loader:
                        if id in ids:
                            import networkx as nx
                            import matplotlib.pyplot as plt
                            print(id)
                            ids.remove(id)
                            for datum in data.to_data_list():
                                g = torch_geometric.utils.to_networkx(datum, to_undirected=True)
                                nx.draw(g)
                                plt.show()

                        optimizer.zero_grad()
                        #print(data.x)
                        pred = model(data.to(device))
                        #print(pred)
                        apart = loss_func(
                            pred[0::2],
                            pred[1::2],
                            torch.tensor([-1] * (len(pred) // 2)).to(device),
                        )
                        together = loss_func(
                            torch.cat((pred[0::4], pred[1::4])),
                            torch.cat((pred[2::4], pred[3::4])),
                            torch.tensor([1] * (len(pred) // 2)).to(device),
                        )
                        # print(apart, together)
                        a = args.loss_parameter
                        if args.loss == "CosineEmbeddingLoss":
                            loss = a*apart+(1-a)*together
                        elif args.loss == "nt_bxent_loss":
                            loss = nt_bxent_loss(pred,
                                torch.tensor([(2*a, 2*b) for a in range(len(pred)//2) for b in range(len(pred)//2)] +
                                [(2*a+1, 2*b+1) for a in range(len(pred)//2) for b in range(len(pred)//2)]).to(device),
                                                 0.5, device)
                        loss.backward()
                        optimizer.step()
                        loss_all += len(pred) / 2 * loss.item()
                    #if _%5 == 0:
                    #    print(loss_all)  # TODO remove
                    #for name, param in model.named_parameters():
                    #    if param.requires_grad:
                    #        print(name, param.data)
                    #        break
                    loss_all /= NUM_RELABEL
                    logger.info(f"Loss: {loss_all}")
                    if loss_all < LOSS_THRESHOLD:
                        logger.info("Early Stop Here")
                        break
                    scheduler.step(loss_all)
                epochs += _

                model.eval()
                T_square_traintest = T2_calculation(dataset_traintest, True)
                T_square_reliability = T2_calculation(dataset_reliability, True)
                print(T_square_traintest, T_square_reliability)

                if (T_square_traintest > THRESHOLD and T_square_reliability < THRESHOLD
                        and not torch.isclose(T_square_traintest, T_square_reliability, atol=EPSILON_CMP)):
                    break
            time_end = time.process_time()
            time_cost = round(time_end - time_start, 2)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")
            #print(isomorphic_flag, reliability_flag, T_square_traintest, T_square_reliability)

            #save to file here
            store.append((part_name, id, isomorphic_flag, T_square_traintest, reliability_flag, T_square_reliability,
                          test_count+1, time_cost, epochs))

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

        store.append(args)
        with open(file, 'ab') as f:
            pickle.dump(store, f)
        store = []

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{args.root}/{args.name_tag}_show.log", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\tnum_layers\thidden_units\tnum_runs\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.num_layers}\t{args.hidden_units}\t{args.num_runs}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    OUT_PATH = "result_BREC_no4v_60cfi"
    NAME = f"h={args.h}_layer={args.layers}_{args.node_label}_{args.use_rd}"
    DATASET_NAME = f"h={args.h}_{args.node_label}"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, device=device)
    #model = get_model(args, device)
    evaluation(dataset, device, args)


if __name__ == "__main__":
    main()
