from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, scatter, softmax, one_hot


class CusRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, head=4, aggr='mean', negative_slope=0.2, dropout=0.2,
                 **kwargs):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.heads = head

        # Define layers
        self.lin = Linear(in_channels, head * out_channels, bias=False, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, head, out_channels))
        self.weight = Parameter(torch.empty(num_relations, out_channels, out_channels))
        self.root_weight = Parameter(torch.empty(out_channels, out_channels))

        # Optional bias
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.att)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):
        # x: (num_nodes, C, F)
        # edge_index: (2, num_edges)
        # edge_type: (num_edges)

        node_nums = x.shape[0]
        x_l = rearrange(self.lin(rearrange(x, 'n c f -> n f c')), 'n f (h o) -> n h o f', h=self.heads)
        # x_r = rearrange(self.lin(rearrange(x, 'n c f -> n f c')), 'n f (h o) -> n h o f', h=self.heads)
        alpha = self.edge_updater(edge_index, x=(x_l, x_l), edge_attr=None)

        out = self.propagate(edge_index, x=(x_l, x_l), edge_type=edge_type, alpha=alpha)
        x_r = x_l.mean(dim=1)
        x_r = rearrange(x_r, 'n o f -> (n f) o')
        out = out + rearrange(x_r @ self.root_weight, '(n f) o ->n o f', n=node_nums)

        if self.bias is not None:
            out += self.bias.unsqueeze(-1)

        return out

    def edge_update(self, edge_index_i, edge_index_j, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att.unsqueeze(-1)).sum(dim=-2)  # (edge,head)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, edge_index, edge_index_j, edge_index_i, edge_type, alpha):
        # ======relation wise======
        x_j = (x_j * alpha.unsqueeze(-2)).mean(dim=1)
        weight = self.weight[edge_type]  # (edge,in c, out c)
        return torch.bmm(rearrange(x_j, 'n o f -> n f o'), weight).permute(0, 2, 1).contiguous()

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
        norm = one_hot(edge_type, self.num_relations, dtype=inputs.dtype)
        norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
        norm = torch.gather(norm, 1, edge_type.view(-1, 1))
        norm = 1. / norm.clamp_(1.)
        inputs = norm.unsqueeze(-1) * inputs
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)


def init_edge(img_per_batch: int, num_batches: int, device: str) -> tuple[Tensor, Tensor]:
    edge_index = []
    edge_type = []

    for batch in range(num_batches):
        start_idx = batch * (2 * img_per_batch + 1)
        total_node_num = start_idx + 2 * img_per_batch + 1
        first_flow_node = start_idx + img_per_batch

        for img_node in range(start_idx, first_flow_node):
            if img_node < first_flow_node - 1:
                edge_index.append([img_node, img_node + 1])
                edge_index.append(
                    [first_flow_node + (img_node - start_idx), first_flow_node + (img_node - start_idx) + 1])
                edge_type.extend([0, 1])  # type 0 img point to the next img; type 1 flow point to the next flow

            edge_index.append([img_node, first_flow_node + (img_node - start_idx)])
            edge_index.append([first_flow_node + (img_node - start_idx), img_node])
            edge_type.extend([2, 2])  # type 2 implies img pointing to flow bilaterally

            edge_index.append([img_node, total_node_num - 1])
            edge_index.append([total_node_num - 1, img_node])
            edge_type.extend([3, 3])  # type 3 img to text bilaterally

            edge_index.append([first_flow_node + (img_node - start_idx), total_node_num - 1])
            edge_index.append([total_node_num - 1, first_flow_node + (img_node - start_idx)])
            edge_type.extend([4, 4])  # type 4 flow to text bilaterally

        # ---- Add global node ----
        global_node_idx = total_node_num  # Assign a new global node index
        for node in range(start_idx, total_node_num):
            # Create edges between global node and all other nodes
            edge_index.append([global_node_idx, node])  # global_node -> node
            edge_index.append([node, global_node_idx])  # node -> global_node
            edge_type.extend([5, 5])  # Define a new edge type for global node connections

    # Convert to torch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type, dtype=torch.long).view(-1).contiguous().to(device)

    # Add self-loops for all nodes including global nodes
    edge_index, _ = add_self_loops(edge_index, num_nodes=int(edge_index.max()) + 1)
    new_edge_type = torch.full((edge_index.size(1) - edge_type.size(0),), int(edge_type.max()) + 1,
                               dtype=torch.long).to(device)

    edge_type = torch.cat([edge_type, new_edge_type], dim=0)

    return edge_index, edge_type


class MultimodalGNN(nn.Module):
    def __init__(self, input_dim, num_relations, num_head, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.rgcn1 = CusRGCNConv(input_dim, input_dim * 2, num_relations, head=num_head)
        self.rgcn2 = CusRGCNConv(input_dim * 2, input_dim, num_relations, head=num_head)
        self.in_channels = input_dim
        self.out_channels = input_dim

        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.BatchNorm2d(self.out_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

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

    def forward(self, x):  # x:[img+flow+word_length,h,w,c]

        # vision & motion( bs frames c h w), text:(bs frames c l)
        (vision_feat, vision_pos), (motion_feat, motion_pos), (text_word_features, text_pos) = x
        device = vision_feat.device
        bs, frames, c, h, w = vision_feat.shape
        text_word_feats, text_word_poses = self.reshape_text_feat_and_pos((text_word_features, text_pos), (h, w),
                                                                          device)
        x_feat = []
        for i in range(bs):
            x_feat_i = torch.cat((vision_feat[i], motion_feat[i],
                                  rearrange(text_word_feats[i], '(t c) h w->t c h w', t=1)), dim=0)
            x_feat_i = rearrange(x_feat_i, ' n c h w ->n c (h w)')
            x_pos_i = torch.cat([vision_feat[i], motion_feat[i],
                                 rearrange(text_word_feats[i], '(t c) h w->t c h w', t=1)], dim=0)
            x_pos_i = rearrange(x_pos_i, ' n c h w ->n c (h w)')

            global_x = torch.mean(x_feat_i + x_pos_i, dim=0, keepdim=True)
            x_feat_i = torch.cat([x_feat_i + x_pos_i, global_x], dim=0)
            x_feat.append(x_feat_i)

        x_feat = rearrange(torch.stack(x_feat, dim=0).to(device), 'b n c f -> (b n) c f')

        # x_feat = [torch.cat((vision_feat[i], motion_feat[i],
        #                      rearrange(text_word_feats[i], '(t c) h w->t c h w', t=1)), dim=0) for i in range(bs)]
        # x_feat = rearrange(torch.stack(x_feat, dim=0).to(device), 'b n c h w -> (b n) c (h w)')
        # x_pos = [torch.cat((vision_pos[i], motion_pos[i],
        #                     rearrange(text_word_poses[i], '(t c) h w->t c h w', t=1)), dim=0) for i in range(bs)]
        # x_pos = rearrange(torch.stack(x_pos, dim=0).to(device), 'b n c h w -> (b n) c (h w)')

        edge_index, edge_type = init_edge(frames, bs, device)
        # x_feat = x_feat + x_pos

        x_feat = self.rgcn1(x_feat, edge_index, edge_type)
        x_feat = self.rgcn2(x_feat, edge_index, edge_type)

        x_feat = rearrange(x_feat, 't c (h w) -> t c h w', h=h)
        vision, motion, global_x = [], [], []
        for i in range(bs):
            start_index = i * 2 * frames + i
            vision_end_index = start_index + frames
            motion_end_index = vision_end_index + frames
            global_x_index = motion_end_index + 2
            vision.append(x_feat[start_index:vision_end_index])
            motion.append(x_feat[vision_end_index:vision_end_index + frames])
            global_x_i = x_feat[global_x_index]
            global_x.append(global_x_i)
        vision = torch.stack(vision, dim=0).to(device)
        motion = torch.stack(motion, dim=0).to(device)
        text_f = torch.stack(global_x, dim=0).to(device)

        feat = rearrange(vision_feat + self.dropout(vision), 'b t c h w -> (b t) c h w')

        # feat = (
        #     self.dropout(
        #         self.relu(
        #             self.norm(
        #                 self.conv(rearrange(vision + vision_feat, 'b t c h w -> (b t) c h w'))))))  # (bs*t,c,h,w)
        return feat, text_f


if __name__ == '__main__':
    # 使用示例：
    height, width = 6, 8
    img_per_batch = 5
    batch = 2
    model = MultimodalGNN(256, 7, 4)
    (vision_feat, vision_pos) = (
        torch.rand(batch, img_per_batch, 256, height, width), torch.rand(batch, img_per_batch, 256, height, width))
    (motion_feat, motion_pos) = (
        torch.rand(batch, img_per_batch, 256, height, width), torch.rand(batch, img_per_batch, 256, height, width))
    (text_feat, text_pos) = (torch.rand(batch, 256, 14), torch.rand(batch, 256, 14))

    out = model(((vision_feat, vision_pos), (motion_feat, motion_pos), (text_feat, text_pos)))
    print(out[0].shape)
    print(out[1].shape)
    t = out[1].unsqueeze(0).expand(4, -1, -1, -1, -1)
    # x 的形状: (l, b, c, h, w)
    print(t.shape)

    # Step 1: 先将 x 的形状从 (l, b, c, h, w) 转换为 (b, l, c, h, w)
    x = t.permute(1, 0, 2, 3, 4)  # 形状变为 (b, l, c, h, w)
    print(x.shape)
    b, l, c, h, w = x.shape
    x = x.view(b, l, c, h * w)
    print(x.shape)

    # Step 2: 展开 l 维度，并在 h, w 维度上进行 cat 操作
    # 在 h 维度上拼接所有层的特征
    x_cat = torch.cat([x[:, i, :, :] for i in range(l)], dim=-1)  # 在最后一个维度拼接，dim=-1
    print(x_cat.shape)
