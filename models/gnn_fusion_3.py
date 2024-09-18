import random
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MultimodalGNN(MessagePassing):

    def __init__(self, feature_dim, num_relations, num_head, negative_slope: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = feature_dim
        self.out_channels = feature_dim
        self.num_relations = num_relations
        self.num_head = num_head
        # self.weight = nn.Parameter(torch.empty(num_relations, self.in_channels, self.out_channels))
        self.negative_slope = negative_slope


        self.dropout = nn.Dropout(p=0.2)

        self.lin = Linear(self.in_channels, num_head * self.out_channels, bias=False,
                          weight_initializer='glorot')
        self.att_src = Parameter(torch.empty(1, num_head, self.out_channels))
        self.att_dst = Parameter(torch.empty(1, num_head, self.out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def message(self, x_j: Tensor, alpha, edge_type: Tensor, edge_index_j: Tensor) -> Tensor:
        out = alpha.unsqueeze(-1) * (x_j.permute(1, 0, 2).contiguous())
        out = out.mean(dim=1)
        # weight = self.weight
        # out = torch.bmm(out.unsqueeze(-2), weight[edge_type]).squeeze(
        #     -2)  # torch.Size([22430, 1, 256]) bmm torch.Size([22430, 256,256])
        return out

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)

    def reshape_text_feat_and_pos(self, text_word_features: Tuple[Tensor, Tensor], size: Tuple[int, int],
                                  device='cpu') -> Tuple[Tensor, Tensor]:
        features, positions = text_word_features
        bs, c, len = features.shape
        h, w = size
        target_size = h * w
        text_word_feats = []
        text_word_poses = []

        for i in range(bs):
            text_word_feat = features[i]
            text_word_pos = positions[i]

            if len < target_size:
                repeat_times = (target_size + len - 1) // len
                text_word_feat = text_word_feat.repeat(1, repeat_times)[:, :target_size]
                text_word_pos = text_word_pos.repeat(1, repeat_times)[:, :target_size]
            elif len > target_size:
                text_word_feat = text_word_feat[:, :target_size]
                text_word_pos = text_word_pos[:, :target_size]

            text_word_feats.append(text_word_feat)
            text_word_poses.append(text_word_pos)

        text_word_feats = torch.stack(text_word_feats, dim=0).view(bs, c, h, w).contiguous().to(device)
        text_word_poses = torch.stack(text_word_poses, dim=0).view(bs, c, h, w).contiguous().to(device)

        return text_word_feats, text_word_poses

    def forward(self, feat):  # x:[img+flow+word_length,h,w,c]

        # vision & motion( bs frames c h w), text:(bs frames c l) bs*frames c hw
        (vision_feat, vision_pos), (motion_feat, motion_pos), (text_word_features, text_pos) = feat
        bs, frames, c, h, w = vision_feat.shape
        device = vision_feat.device

        vf = rearrange(vision_feat + vision_pos, 'b t c h w -> b (t h w) c')
        mf = rearrange(motion_feat + motion_pos, 'b t c h w -> b (t h w) c')
        tf = rearrange(text_word_features + text_pos, 'b c l -> b l c')

        res = []
        for i in range(bs):
            edge_index, edge_type = create_sparse_edges_and_edge_types(h, w, tf.shape[1], device=device)
            feat = torch.cat((vf[i], mf[i], tf[i]), dim=0).to(device)
            x_src = x_dst = self.lin(feat).view(-1, self.num_head, c).contiguous()
            x = x_src.permute(1, 0, 2).contiguous()

            alpha_src = (x_src * self.att_src).sum(dim=-1)  # (11,head,output channel) * (1,head,output channel)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)  # (11,1,256) * (1,1,256)
            alpha = (alpha_src, alpha_dst)
            alpha = self.edge_updater(edge_index=edge_index, alpha=alpha, edge_attr=None,
                                      size=None)

            out = self.propagate(edge_index=edge_index, x=x, alpha=alpha, edge_type=edge_type,
                                 size=(x_src.shape[0], x_src.shape[0]))

            vision_ = rearrange(out[:frames * h * w], '(t h w) c -> t c h w', h=h, w=w)

            res.append(vision_)
        return rearrange(torch.stack(res, dim=0), 'b t c h w -> (b t) c h w').contiguous().to(device)

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=0.2, training=self.training)
        return alpha  # (edge_num, nhead)


def create_sparse_edges_and_edge_types(H, W, L, num_random_edges=16, neighbor_range=3, block_size=16, jump=6,
                                       device='cpu'):
    num_nodes = 2 * H * W + L
    edges = []
    edge_types = []

    def add_bidirectional_edge(edges, edge_types, node_a, node_b, edge_type):
        edges.append([node_a, node_b])
        edges.append([node_b, node_a])
        edge_types.append(edge_type)
        edge_types.append(edge_type)

    # Function to add bidirectional edges within a range in a region with jumps
    def add_neighbor_edges(start_idx, H, W, edge_type, neighbor_range, jump):
        for i in range(0, H, jump):
            for j in range(0, W, jump):
                node_idx = start_idx + i * W + j
                for di in range(-neighbor_range, neighbor_range + 1):
                    for dj in range(-neighbor_range, neighbor_range + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and (di != 0 or dj != 0):
                            neighbor_idx = start_idx + ni * W + nj
                            add_bidirectional_edge(edges, edge_types, node_idx, neighbor_idx, edge_type)

    # Function to add random bidirectional edges between blocks in a region and l
    def add_random_edges_between_blocks(region_start_idx, H, W, l_start_idx, L, edge_type, num_edges, block_size):
        blocks = [(i, j) for i in range(0, H, block_size) for j in range(0, W, block_size)]
        l_nodes = list(range(l_start_idx, l_start_idx + L))
        for (bi, bj) in blocks:
            block_nodes = [region_start_idx + (bi + di) * W + (bj + dj) for di in range(block_size) for dj in
                           range(block_size) if bi + di < H and bj + dj < W]
            for _ in range(num_edges):
                node_a = random.choice(block_nodes)
                node_b = random.choice(l_nodes)
                add_bidirectional_edge(edges, edge_types, node_a, node_b, edge_type)

    # First h * w region with edge type 0
    add_neighbor_edges(0, H, W, 0, neighbor_range, jump)

    # Second h * w region with edge type 1
    offset = H * W
    add_neighbor_edges(offset, H, W, 1, neighbor_range, jump)

    # Between first and second h * w with edge type 2
    for i in range(H * W):
        add_bidirectional_edge(edges, edge_types, i, offset + i, 2)

    # Between first h * w and l with edge type 3
    add_random_edges_between_blocks(0, H, W, 2 * H * W, L, 3, num_random_edges, block_size)

    # Between second h * w and l with edge type 4
    add_random_edges_between_blocks(offset, H, W, 2 * H * W, L, 4, num_random_edges, block_size)

    # l region with edge type 5
    for i in range(L):
        for j in range(i + 1, L):
            add_bidirectional_edge(edges, edge_types, 2 * H * W + i, 2 * H * W + j, 5)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = (torch.tensor(edge_types, dtype=torch.long).view(-1).contiguous()
                 .to(device))  # ensure edge_type is a flat tensor
    return edge_index, edge_type


if __name__ == '__main__':
    model = MultimodalGNN(256, 6, 8)
    (vision_feat, vision_pos) = (torch.rand(1, 5, 256, 48, 64), torch.rand(1, 5, 256, 48, 64))
    (motion_feat, motion_pos) = (torch.rand(1, 5, 256, 48, 64), torch.rand(1, 5, 256, 48, 64))
    (text_feat, text_pos) = (torch.rand(1, 256, 14), torch.rand(1, 256, 14))
    out = model(((vision_feat, vision_pos), (motion_feat, motion_pos), (text_feat, text_pos)))
    print(out.shape)
