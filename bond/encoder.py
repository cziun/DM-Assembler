import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class Encoder(nn.Module):
    def __init__(self, dim_in, num_edge_type, dim_hidden, dim_out, t=4):
        super(Encoder, self).__init__()
        # self.embedding = Embedding(one_hot_sizes, dim_embeddings)
        self.num_edge_type = num_edge_type
        self.t = t  # number of iterations
        self.node_trans = nn.Linear(dim_in, dim_hidden) 
        # self.edge_trans = nn.Embedding(num_edge_type, dim_hidden)
        self.edge_trans = nn.Linear(num_edge_type, dim_hidden)
        self.conv = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        # self.conv1 = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        # self.conv2 = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        self.linear = nn.Linear(dim_hidden * self.t, dim_out)

    def embed_node(self, x, edge_index, edge_attr):
        x = self.node_trans(x.float())  # [total_num_nodes, dim_hidden]
        edge_attr = self.edge_trans(edge_attr.float()).squeeze(1)  # [total_num_edges, dim_hidden]
        all_x = []
        for _ in range(self.t):
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            all_x.append(x)
        all_x = torch.cat(all_x, dim=-1)  # [total_num_nodes, dim_hidden * t]
        return x, all_x

    def embed_graph(self, all_x, graph_ids, node_mask=None):
        res = torch.zeros((graph_ids[-1] + 1, all_x.shape[-1]), device=all_x.device)  # [num_graphs, dim_out]
        if node_mask is not None:
            graph_ids, all_x = graph_ids[~node_mask], all_x[~node_mask]
        res.index_add_(0, graph_ids, all_x)
        res = self.linear(res)  # to dim out
        return res

    def forward(self, batch, return_x=False):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x, all_x = self.embed_node(x, edge_index, edge_attr)
        res = torch.zeros((batch.num_graphs, all_x.shape[-1]), device=all_x.device)  # [num_graphs, dim_out]
        res.index_add_(0, batch.batch, all_x)
        res = self.linear(res)  # to dim out

        if return_x:
            return res, x
        return res


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_func, num_layers):
        super(MLP, self).__init__()
        assert num_layers > 0
        if num_layers == 1:
            self.seq = nn.Linear(dim_in, dim_out)
        else:
            seq = [nn.Linear(dim_in, dim_hidden), act_func()]
            for i in range(num_layers - 2):
                seq.append(nn.Linear(dim_hidden, dim_hidden))
                seq.append(act_func())
            seq.append(nn.Linear(dim_hidden, dim_out))
            self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)
    