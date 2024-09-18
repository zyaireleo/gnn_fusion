from typing import Union, Tuple, Optional

import sympy
import torch
from einops import rearrange
from torch import nn, Tensor
from torch_geometric.nn import RGCNConv, MessagePassing, FastRGCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops, one_hot, scatter
import torch.nn.functional as F
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


class SimplifiedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(SimplifiedMultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

        batch_size, seq_length, embed_dim = query.size()

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if attn_mask is not None:
            attn_scores += attn_mask.unsqueeze(1).unsqueeze(2)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Apply layer normalization
        output = self.norm(query + attn_output)

        if need_weights:
            attn_weights = attn_probs.mean(dim=1) if average_attn_weights else attn_probs
        else:
            attn_weights = None

        return output, attn_weights


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = SimplifiedMultiHeadSelfAttention(d_model, nhead, dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=tgt + query_pos,
                                   key=memory + pos,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt


def unzip_feature(feature: Tuple[Tensor, Tensor], edge_index: Tensor, device: str):
    x_feat, x_pos = feature
    x_j = torch.index_select(x_feat, 0, edge_index[0]).to(device)  # 出度节点 Tensor
    x_j_pos = torch.index_select(x_pos, 0, edge_index[0]).to(device)  # 出度节点位置编码
    x_i = torch.index_select(x_feat, 0, edge_index[1]).to(device)  # 入度节点 Tensor
    x_i_pos = torch.index_select(x_pos, 0, edge_index[1]).to(device)
    return x_feat, x_i, x_i_pos, x_j, x_j_pos


class MultimodalGNN(nn.Module):
    def __init__(self, feature_dim, num_relations, num_head, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = feature_dim
        self.out_channels = feature_dim
        self.num_relations = num_relations
        self.alpa = nn.Embedding(num_relations, feature_dim)
        self.num_head = num_head
        self.linear_edge = nn.Linear(1, self.out_channels, False)
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.out_channels))

        self.conv = nn.Conv2d(in_channels=2 * self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.BatchNorm2d(self.out_channels)

        self.relation_cross_attention = nn.ModuleList()
        for _ in range(self.num_relations):
            self.relation_cross_attention.append(CrossAttentionLayer(
                d_model=feature_dim,
                nhead=self.num_head // 2,
                dropout=0.2
            ))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reshape_text_feat_and_pos(self, text_word_features: Tuple[Tensor, Tensor], size: Tuple[int, int],
                                  device='cpu') \
            -> Tuple[Tensor, Tensor]:
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
                pad_size = target_size - len
                text_word_feat = torch.cat(
                    [text_word_feat, torch.zeros(c, pad_size, dtype=text_word_feat.dtype, device=device)], dim=1)
                text_word_pos = torch.cat(
                    [text_word_pos, torch.zeros(c, pad_size, dtype=text_word_pos.dtype, device=device)], dim=1)
            elif len > target_size:
                text_word_feat = text_word_feat[:, :target_size]
                text_word_pos = text_word_pos[:, :target_size]
            text_word_feats.append(text_word_feat)
            text_word_poses.append(text_word_pos)

        text_word_feats = torch.stack(text_word_feats, dim=0).view(bs, c, h, w).contiguous().to(device)
        text_word_poses = torch.stack(text_word_poses, dim=0).view(bs, c, h, w).contiguous().to(device)
        return text_word_feats, text_word_poses

    def forward(self, x):  # x:[img+flow+word_length,h,w,c]

        # vision & motion( bs frames c h w), text:(bs frames c l)
        (vision_feat, vision_pos), (motion_feat, motion_featmotion_pos), (text_word_features, text_pos) = x

        vision_feat = rearrange(vision_pos, 'b t c h w -> (b t) c h w')
        motion_feat = rearrange(motion_feat, 'b t c h w -> (b t) c h w')
        feat = torch.cat([vision_feat, motion_feat], dim=1)

        feat = self.dropout(self.relu(self.norm(self.conv(feat))))  # (bs*t,c,h,w)
        return feat

    def graph_wise_propagation_and_aggregation(self, x: Tuple[Tensor, Tensor],
                                               edge_index: Tensor,
                                               edge_type: Tensor, device: str):
        x_feat, x_pos = x
        node_nums = len(x_feat)
        edge_attr = self.linear_edge(torch.ones(size=(node_nums, 1), device=device))  # (11,256)
        alpha = (edge_attr @ self.weight)  # (11,256)
        alpha = alpha[edge_index[0]]  # (edge_num,256)
        alpha = softmax(src=alpha, index=edge_index[1], num_nodes=node_nums)
        alpha = F.dropout(alpha, p=0.2, training=self.training)  # (edge_num,256)
        c, h, w = x_feat[0].shape
        node_features = ((alpha.unsqueeze(-1) * x_feat.view(node_nums, c, -1)[edge_index[0]])
                         .contiguous())  # (edge_num,c,h*w)
        node_features = (scatter(node_features, edge_index[1], dim=0, reduce='sum')
                         .view(node_nums, c, h, w).contiguous())  # (node_num,c,h,w)
        return x_feat + node_features, x_pos

    def relation_propagate_and_aggregation(self, x: Tuple[Tensor, Tensor],
                                           edge_index_tmp: Tensor, cross_attention: nn.Module, device: str) -> Tensor:
        x_feat, x_i, x_i_pos, x_j, x_j_pos = unzip_feature(x, edge_index_tmp, device)
        node_num, c, h, w = x_i.shape if len(x_i.shape) == 4 else x_j.shape

        out = cross_attention(tgt=x_i.view(-1, node_num, c).contiguous(), memory=x_j.view(-1, node_num, c).contiguous(),
                              pos=x_j_pos.view(-1, node_num, c).contiguous(),
                              query_pos=x_i_pos.view(-1, node_num, c).contiguous())

        out = out.view(node_num, c, h, w).contiguous()
        # 和原来直接相加
        for node_index, node_new_val in zip(edge_index_tmp[1], out):
            x_feat[node_index] = node_new_val + x_feat[node_index]
        return x_feat


def init_edge(img_per_batch: int, device: str) -> Tuple[Tensor, Tensor]:
    edge_index = []
    edge_type = []
    total_node_num = 2 * img_per_batch + 1
    first_flow_node = img_per_batch

    for img_node in range(first_flow_node):
        if img_node < first_flow_node - 1:
            edge_index.append([img_node, img_node + 1])
            edge_index.append([first_flow_node + img_node, first_flow_node + img_node + 1])
            edge_type.append([0, 1])  # type 0 img point to the next img;type 1 flow point to the next flow
        edge_index.append([img_node, first_flow_node + img_node])
        edge_index.append([first_flow_node + img_node, img_node])
        edge_type.append([2, 2])  # type 2 implies img pointing to flow bilaterally

        edge_index.append([img_node, total_node_num - 1])
        edge_index.append([total_node_num - 1, img_node])
        edge_type.append([3, 3])  # type 3 img to text bilaterally

        edge_index.append([first_flow_node + img_node, total_node_num - 1])
        edge_index.append([total_node_num - 1, first_flow_node + img_node])
        edge_type.append([4, 4])  # type 4 flow to text bilaterally

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_type = (torch.tensor(edge_type, dtype=torch.long).view(-1).contiguous()
                 .to(device))  # ensure edge_type is a flat tensor

    return edge_index, edge_type


if __name__ == '__main__':
    edge_index, edge_type = init_edge(5, 'cpu')
    model = MultimodalGNN(256, 5, 8)
    (vision_feat, vision_pos) = (torch.rand(2, 5, 256, 48, 64), torch.rand(2, 5, 256, 48, 64))
    (motion_feat, motion_pos) = (torch.rand(2, 5, 256, 48, 64), torch.rand(2, 5, 256, 48, 64))
    (text_feat, text_pos) = (torch.rand(2, 256, 14), torch.rand(2, 256, 14))
    out = model(((vision_feat, vision_pos), (motion_feat, motion_pos), (text_feat, text_pos)))
    print(out.shape)
