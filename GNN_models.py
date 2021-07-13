import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GCNConv, SAGEConv
"""
Implementing diffpool!
Yonsei App.Stat. Sunwoo Kim
"""

class diffpool_gnn(torch.nn.Module) :

    def __init__(self, dataset, gconv = SAGEConv, device = "cuda", latent_dim = [16, 16, "d", 16, 16],
                 diff_dim = [10, 10], end_dim = [16, 6]) :
        super(diffpool_gnn, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.device = device
        self.latent = latent_dim

        # First, define diffpooling layer
        self.diff_gnn_emb = gconv(in_channels = latent_dim[1],
                                       out_channels = diff_dim[0])
        self.diff_gnn_pool = gconv(in_channels = latent_dim[1],
                                       out_channels = diff_dim[1])
        self.final_dif = gconv(in_channels=latent_dim[-1],
                               out_channels=end_dim[0])
        self.final_pool = gconv(in_channels=latent_dim[-1],
                               out_channels=1)

        self.convs.append(
            gconv(in_channels=dataset.num_features,
                  out_channels=latent_dim[0])
        )

        for i in range(0, (len(latent_dim) - 1)) :
            if latent_dim[i+1] == "d" :
                self.convs.append(gconv(in_channels=latent_dim[i],
                                        out_channels=diff_dim[0]))
            elif latent_dim[i] == "d" :
                self.convs.append(gconv(in_channels=diff_dim[0],
                                        out_channels=latent_dim[i + 1]))
            else :
                self.convs.append(gconv(in_channels=latent_dim[i],
                                        out_channels=latent_dim[i+1]))
        self.read_out = torch.nn.Linear(end_dim[0], end_dim[1])

    def reset_parameters(self) :
        for conv_layer in self.convs :
            conv_layer.reset_parameters()
        self.diff_gnn_pool.reset_parameters()
        self.diff_gnn_emb.reset_parameters()
        self.final_dif.reset_parameters()
        self.final_pool.reset_parameters()
        self.read_out.reset_parameters()

    def forward(self, data) :
        self.layer_h = []
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs) :

            if i == 2 : # If it is diff pooling
                tup = (edge_index[0],
                       edge_index[1])
                A = torch.zeros(x.shape[0], x.shape[0]).to(self.device)
                A[tup] = 1
                self.A1 = A
                side_z = torch.relu(self.diff_gnn_emb(x = x,
                                           edge_index = edge_index)) # dim = node_n x k
                side_s = torch.softmax(self.diff_gnn_pool(x = x,
                                           edge_index = edge_index), dim = 1)
                self.S1 = side_s
                x = torch.matmul(torch.transpose(side_s, 1, 0),
                                 side_z) # new_n x k
                A = torch.matmul(torch.matmul(torch.transpose(side_s, 1, 0), A), side_s) # new_n x new_n
                edge_index = torch.vstack([torch.where(A > 0)[0], torch.where(A > 0)[1]])
            else : # Not diff-pooling
                x = torch.relu(conv(x=x,
                                    edge_index=edge_index))

        self.A2 = A
        final_z = torch.relu(self.final_dif(x = x,
                                            edge_index = edge_index))
        final_s = torch.softmax(self.final_pool(x = x,
                                                edge_index = edge_index), dim = 1)
        self.S2 = final_s
        X = torch.matmul(torch.transpose(final_s, 1, 0),
                         final_z)
        y = self.read_out(X)
        return y

    def frobenious_norm(self) :
        L1 = torch.sqrt(torch.sum((self.A1 - torch.matmul(self.S1, torch.transpose(self.S1, 0, 1))) ** 2))
        L2 = torch.sqrt(torch.sum((self.A2 - torch.matmul(self.S2, torch.transpose(self.S2, 0, 1))) ** 2))
        return L1 + L2

    def cross_entropy(self) :
        h1 = torch.sum(-self.S1*torch.log2(self.S1+1e-8))
        h2 = torch.sum(-self.S2*torch.log2(self.S2+1e-8))
        return h1 + h2