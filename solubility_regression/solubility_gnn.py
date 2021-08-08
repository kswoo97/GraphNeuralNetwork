# 가장 무난한 Neural Message Passing Conv를 사용했습니다.
# 모델 자체에 Edge Attribute를 쓸 수 있기때문에 채택했습니다.

import torch
from torch_geometric.nn import NNConv, GCNConv, SAGEConv


class NMP_Conv(torch.nn.Module):
    def __init__(self, dataset, gconv=NNConv, latent_dim=[8, 8], device = "cpu"):
        super(NMP_Conv, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.edge_linear = torch.nn.ModuleList()
        self.device = device
        self.last_dim = latent_dim[-1]

        self.edge_linear.append(
            torch.nn.Linear(
                dataset.num_edge_features, dataset.num_features * latent_dim[0]
            )
        )
        for lin in range((len(latent_dim) - 1)):
            self.edge_linear.append(
                torch.nn.Linear(
                    dataset.num_edge_features, latent_dim[lin] * latent_dim[lin + 1]
                )
            )

        self.convs.append(
            gconv(dataset.num_features, latent_dim[0], nn=self.edge_linear[0])
        )
        for i in range(0, (len(latent_dim) - 1)):
            self.convs.append(
                gconv(latent_dim[i], latent_dim[i + 1], nn=self.edge_linear[i + 1])
            )

        self.last_linear = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self):
        # Reset linear / convolutional parameters
        for lin_layer in self.edge_linear:
            lin_layer.reset_parameters()
        for conv_layer in self.convs:
            conv_layer.reset_parameters()
        self.last_linear.reset_parameters()

    def forward(self, data, training_with_batch):
        if training_with_batch : # Batch training
            x, edge_index, edge_att = data.x, data.edge_index, data.edge_attr
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index, edge_att))
            outs = torch.zeros((1, self.last_dim), requires_grad=True).to(self.device)
            for i in range(data.batch.unique().shape[0]) :
                idx = torch.where(data.batch == i)[0]
                batch_sum = torch.sum(x[idx, :], 0)
                outs = torch.vstack([outs, batch_sum])
            outs = outs[1:, :]
            x = self.last_linear(outs)
        else : # Not batchwise training
            x, edge_index, edge_att = data.x, data.edge_index, data.edge_attr
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index, edge_att))
            x = torch.sum(x, 0)
            x = self.last_linear(x)
        return x