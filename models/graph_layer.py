from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GraphLayer(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self._alpha = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.att_i = Parameter(torch.empty(1, heads, out_channels))
        self.att_j = Parameter(torch.empty(1, heads, out_channels))
        self.att_em_i = Parameter(torch.empty(1, heads, out_channels))
        self.att_em_j = Parameter(torch.empty(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        embedding: Optional[Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:

        # Handle bipartite / homogeneous node input representations
        if isinstance(x, Tensor):
            x_src = x_dst = self.lin(x)
        else:
            x_src = self.lin(x[0])
            x_dst = self.lin(x[1]) if x[1] is not None else x_src

        num_nodes = x_dst.size(0)

        # Self-loop transformation
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Message propagation step
        out = self.propagate(
            edge_index=edge_index,
            x=(x_src, x_dst),
            embedding=embedding,
            return_attention_weights=return_attention_weights,
            size=None,
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self._alpha = self._alpha, None
            return out, (edge_index, alpha)

        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        size_i: Optional[int],
        embedding: Optional[Tensor],
        return_attention_weights: bool,
    ) -> Tensor:

        # Reshape for multi-head attention: [E, Heads, Out_Channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        if embedding is not None:
            embedding_i = embedding[edge_index_i].unsqueeze(1)
            embedding_j = embedding[edge_index_j].unsqueeze(1)

            key_i = torch.cat((x_i, embedding_i.expand(-1, self.heads, -1)), dim=-1)
            key_j = torch.cat((x_j, embedding_j.expand(-1, self.heads, -1)), dim=-1)
        else:
            key_i, key_j = x_i, x_j
            cat_att_i, cat_att_j = self.att_i, self.att_j

        # Calculate attention scores
        alpha = (key_i * cat_att_i).sum(dim=-1) + (key_j * cat_att_j).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Normalize attention coefficients using PyG softmax
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)

        if self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if return_attention_weights:
            self._alpha = alpha

        # Weighted message aggregation
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )