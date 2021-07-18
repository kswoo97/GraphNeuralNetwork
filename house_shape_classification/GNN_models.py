import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GCNConv, SAGEConv
"""
Implementing Benzene dataset Classifier
Yonsei App.Stat. Sunwoo Kim
"""

class house_classifier(torch.nn.Module) :

    def __init__(self, dataset, gconv = SAGEConv, device = "cuda", latent_dim = [16, 16, 16]) :
        super(house_classifier, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.device = device
        self.latent = latent_dim

        self.convs.append(
            gconv(in_channels=dataset.num_features,
                  out_channels=latent_dim[0])
        )
        for i in range(0, (len(latent_dim) - 1)) :
            self.convs.append(gconv(in_channels=latent_dim[i],
                                    out_channels=latent_dim[i+1]))
        self.read_out = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self) :
        for conv_layer in self.convs :
            conv_layer.reset_parameters()
        self.read_out.reset_parameters()

    def forward(self, data, training_with_batch) :

        if training_with_batch :
            self.layer_h = []
            x, edge_index = data.x, data.edge_index
            for conv in self.convs :
                x = torch.relu(conv(x = x,
                                    edge_index = edge_index))
                self.layer_h.append(x)
            # batchwise computing
            self.outs = torch.zeros(self.latent[-1], dtype = torch.float, requires_grad=True, device = self.device).reshape(1,-1)
            for i in range(torch.unique(data.batch).shape[0]) :
                indexs = torch.where(data.batch==i)[0]
                x_ = torch.sum(x[indexs], axis = 0).reshape(1, -1)
                self.outs = torch.cat((self.outs, x_), 0)
            self.outs = self.outs[1:, :]
            self.embeddings = self.outs.clone()
            x = self.read_out(self.outs.float())
            return F.sigmoid(x)
        else :
            self.layer_h = []
            x, edge_index = data.x, data.edge_index
            for conv in self.convs :
                x = torch.relu(conv(x = x,
                                    edge_index = edge_index))
                self.layer_h.append(x)
            x_ = torch.sum(x, axis = 0)
            self.embeddings = x_.clone()
            x = self.read_out(x_)
            return F.sigmoid(x)