import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from tqdm import tqdm
"""
Finding house shape data
Yonsei App.Stat. Sunwoo Kim
"""

class BA_house_generator() :

    def __init__(self, max_n, min_n, edge_rate, data_type, r_seed) :

        self.max_n = max_n
        self.min_n = min_n
        self.edge_rate = edge_rate
        self.data_type = data_type
        self.seed = r_seed
        self.nx_list = []
        self.data_list = []

    def random_generator(self, y, seed) :
        '''
        [Top  Middle  Bottom  None]
        :param data_type:
        :param y:
        :param seed:
        :return:
        '''

        if self.data_type == "single_house" :
            if y == 0 : # Build full house
                node_n = int(np.random.choice(np.arange(self.min_n, self.max_n), 1))
                G = nx.barabasi_albert_graph(n=node_n, m = self.edge_rate, seed=seed)
                G.add_edge(int(node_n-1), int(node_n))
                G.add_edge(int(node_n-1), int(node_n+1))
                G.add_edge(int(node_n), int(node_n+1))
                G.add_edge(int(node_n), int(node_n+2))
                G.add_edge(int(node_n+1), int(node_n+3))
                G.add_edge(int(node_n+2), int(node_n+3))
                edges = np.array(G.edges)
                inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                       edges[:, 0].reshape(-1, 1)]) # Changing order to tell its undirected
                edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                          dtype = torch.long)
                x = torch.zeros((int(node_n+4), 4,))
                x[:int(node_n-1), 3] = 1 # Nothing
                x[int(node_n-1), 0] = 1 # Head
                x[int(node_n), 1] = 1 # Middle
                x[int(node_n + 1), 1] = 1 # Middle
                x[int(node_n + 2), 2] = 1 # Bottom
                x[int(node_n + 3), 2] = 1 # Bottom
                data = Data(x = x, edge_index = edge_index.t().contiguous(), y = torch.tensor([1], dtype = torch.float))

            elif y == 1 : # Build a partial house
                node_n = int(np.random.choice(np.arange(self.min_n, self.max_n), 1))
                G = nx.barabasi_albert_graph(n=node_n, m=self.edge_rate, seed=seed)
                G.add_edge(int(node_n - 1), int(node_n))
                G.add_edge(int(node_n - 1), int(node_n + 1))
                G.add_edge(int(node_n), int(node_n + 1))
                edges = np.array(G.edges)
                inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                       edges[:, 0].reshape(-1, 1)])  # Changing order to tell its undirected
                edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                          dtype=torch.long)
                x = torch.zeros((int(node_n + 2), 4))
                x[:int(node_n - 1), 3] = 1  # Nothing
                x[int(node_n - 1), 0] = 1  # Head
                x[int(node_n), 1] = 1  # Middle
                x[int(node_n + 1), 1] = 1  # Middle
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y = torch.tensor([0], dtype = torch.float))

            elif y == 2 : # Do not build a house
                node_n = int(np.random.choice(np.arange(self.min_n, self.max_n), 1))
                G = nx.barabasi_albert_graph(n=node_n, m=self.edge_rate, seed=seed)
                edges = np.array(G.edges)
                inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                       edges[:, 0].reshape(-1, 1)])  # Changing order to tell its undirected
                edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                          dtype=torch.long)
                x = torch.zeros((int(node_n), 4))
                x[:int(node_n - 1), 3] = 1  # Nothing
                x[int(node_n - 1), 0] = 1  # Head
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y = torch.tensor([0], dtype = torch.float))

            else :
                raise TypeError("For single house example, y should be given one of {0, 1, 2}")

        self.nx_list.append([G, y])
        self.data_list.append(data)

    def visualization(self, n):
        if self.data_type == "single_house" :
            G = self.nx_list[n][0]
            y = self.nx_list[n][1]
            if y == 0 :
                node_n = np.array(G.nodes).shape[0]
                color_map = []
                for i in range(int(node_n-5)) :
                    color_map.append("orange")
                color_map.append("red")
                color_map.append("green"); color_map.append("green")
                color_map.append("green") ; color_map.append("green")
                nx.draw(G, node_color=color_map, with_labels=True, pos=nx.kamada_kawai_layout(G))
                plt.show()
            elif y == 1 :
                node_n = np.array(G.nodes).shape[0]
                color_map = []
                for i in range(int(node_n - 3)):
                    color_map.append("orange")
                color_map.append("red")
                color_map.append("green");
                color_map.append("green")
                nx.draw(G, node_color=color_map, with_labels=True, pos=nx.kamada_kawai_layout(G))
                plt.show()
            else :
                node_n = np.array(G.nodes).shape[0]
                color_map = []
                for i in range(int(node_n - 1)):
                    color_map.append("orange")
                color_map.append("red")
                nx.draw(G, node_color=color_map, with_labels=True, pos=nx.kamada_kawai_layout(G))
                plt.show()

    def dataset_generator(self, num_graph) :
        if self.data_type == "single_house" :
            seed = self.seed
            for epoch in tqdm(range(num_graph)) :
                np.random.seed(seed); cur_i = np.random.choice([0, 1], 1)
                if cur_i == 0 : # Generate Full house
                    self.random_generator(y = 0, seed = seed)
                else : # Build negative case
                    np.random.seed(seed); cur_2i= np.random.choice([1, 2], 1)
                    if cur_2i == 1 : # Build partial house
                        self.random_generator(y = 1, seed = seed)
                    else : # Build none house
                        self.random_generator(y = 2, seed = seed)
                seed += 1