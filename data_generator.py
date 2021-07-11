import torch
import numpy as np
import torch.utils.data
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm

class molecular_data_generator(torch.utils.data.Dataset) :

    def __init__(self, x_data, y_data) :
        self.x_data = x_data
        self.y_data = y_data

    def data_generation(self):
        self.out_list = []
        for i, data in tqdm(enumerate(self.x_data)) :
            edge_index = torch.tensor(np.vstack([data["receivers"],
                                                 data["senders"]]).transpose(),
                                      dtype = torch.long)
            x = torch.tensor(data["nodes"], dtype = torch.float)
            edge_feat = torch.tensor(data["edges"],
                                     dtype = torch.float)
            self.out_list.append(Data(x = x,
                 edge_attr=edge_feat,
                 edge_index=edge_index.t().contiguous(),
                 y = torch.tensor(self.y_data[i],
                                  dtype = torch.float)))