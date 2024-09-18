# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor, autocast
from torch.nn.init import xavier_uniform_, constant_, normal_
from torch_geometric.nn import RGCNConv, GATv2Conv

from models.ops.modules import MSDeformAttn
from util.misc import inverse_sigmoid


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_level = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)  # reference point here (x, y)

        self._reset_parameters()
        self.rgcn_conv = RGCNConv(256, 256, 2)
        self.agt_conv = GATv2Conv(256, 256, concat=False)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, tgt, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None
        """
        srcs (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], input of encoder
        tgt (Tensor): [batch_size, time, c, num_queries_per_frame]
        masks (list[Tensor]): list of tensors num_layers x [batch_size*time, hi, wi], the mask of srcs
        pos_embeds (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], position encoding of srcs
        query_embed (Tensor): [num_queries, c]
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # [batch_size, hi*wi, c]
            mask = mask.flatten(1)  # [batch_size, hi*wi]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [batch_size, hi*wi, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)

            src_flatten.append(src)
            mask_flatten.append(mask)

        # For a clip, concat all the features, first fpn layer size, then frame size
        src_flatten = torch.cat(src_flatten, 1)  # [bs*t, \sigma(hi*wi), c]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs*t, \sigma(hi*wi)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)  # torch.Size([4, 2])
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # tensor([   0, 3840, 4800, 5040])
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # torch.Size([5, 4, 2])

        # encoder
        # memory: [bs*t, \sigma(hi*wi), c]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            b, t, q, c = tgt.shape  # tgt torch.Size([1, 5, 5, 256])

            tgt = rearrange(tgt, 'b t q c -> (b t) q c')  # tgt torch.Size([ 5, 5, 256])

            query_embed = query_embed.unsqueeze(0).expand(b * t, -1,
                                                          -1)  # [batch_size*time, num_queries_per_frame, c] :torch.Size([5，5, 256])
            device = query_embed.device
            edge_index, edge_type = initialize_edges_and_types(b, t, 5, device)
            edge_index = edge_index.to(device)
            edge_type = edge_type.to(device)

            out = query_embed.reshape(-1, 256).contiguous()
            out = self.rgcn_conv(out, edge_index, edge_type)
            out = self.agt_conv(out, edge_index)
            query_embed = out.reshape(b * t, q, -1).contiguous()

            reference_points = self.reference_points(
                query_embed).sigmoid()  # [batch_size*time, num_queries_per_frame, 2]
            init_reference_out = reference_points  # [5,5,2]

        # decoder
        hs, inter_references, inter_samples = self.decoder(tgt, reference_points, memory,
                                                               spatial_shapes, level_start_index, valid_ratios,
                                                               query_embed,
                                                               mask_flatten)
        # [4,5,5,256] [4,5,5,2] [4,1,5,30,2]

        inter_references_out = inter_references

        # convert memory to fpn format
        memory_features = []  # 8x -> 32x
        spatial_index = 0
        for lvl in range(self.num_feature_level - 1):
            h, w = spatial_shapes[lvl]
            # [bs*t, c, h, w]
            memory_lvl = (memory[:, spatial_index: spatial_index + h * w, :].reshape(bs, h, w, c).permute(0, 3, 1, 2)
                          .contiguous())
            memory_features.append(memory_lvl)
            spatial_index += h * w

        if self.two_stage:
            return hs, memory_features, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, inter_samples
        # hs: [l, batch_size*time, num_queries_per_frame, c], where l is number of decoder layers(default to 4)[4,5,5,256]
        # init_reference_out: [batch_size*time, num_queries_per_frame, 2] [5,5,2]
        # inter_references_out: [l, batch_size*time, num_queries_per_frame, 4] [4,5,5,4]
        # memory: [batch_size*time, \sigma(hi*wi), c] [5,5100,256]
        # memory_features: list[Tensor]
        # inter_samples: [4,1, 5, 30, 2]

        return hs, memory_features, init_reference_out, inter_references_out, None, None, inter_samples


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, sampling_locations, attention_weights = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                                                                     src, spatial_shapes, level_start_index,
                                                                     padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src  # torch.Size([5, 5100, 256])
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # torch.Size([5, 5100, 4, 2])
        output = output.float()
        pos = pos.float()
        with autocast(device_type='cuda',enabled=False):

            for _, layer in enumerate(self.layers):
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                # output: torch.Size([5, 5100, 256]), pos: torch.Size([5, 5100, 256]), reference_points: torch.Size([5, 5100, 4, 2]),spatial_shapes: torch.Size([4, 2]),
                # level_start_index: tensor([   0, 3840, 4800, 5040]),padding_mask: torch.Size([5, 5100])

            return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        with autocast(device_type='cuda',enabled=False):
            tgt = tgt.float()
            query_pos = query_pos.float()
            src = src.float()
            tgt2, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                                                          reference_points,
                                                                          src, src_spatial_shapes, level_start_index,
                                                                          src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt = self.forward_ffn(tgt)
            assert not torch.isnan(tgt).any(), f"tgt contains NaN: {tgt}"

            return tgt, sampling_locations, attention_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        # we modify here for get the information of sample points
        output = tgt 

        intermediate = []
        intermediate_reference_points = []
        intermediate_samples = []  # sample points
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # reference_points:[5,5,1,2]   src_valid_ratios:[5, 1,4, 2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]  # [5,5,4,2]
            output, sampling_locations, attention_weights = layer(output, query_pos, reference_points_input,
                                                                  src, src_spatial_shapes, src_level_start_index,
                                                                  src_padding_mask)

            assert not torch.isnan(output).any(), f"Reference contains NaN: {output}"

            # sampling_loactions: [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2],
            # output: [5,5,256]
            #                     [B, Q, n_head, n_level(num_feature_level*num_frames), n_points, 2]
            # attention_weights: [B, Q, n_head, n_level(num_feature_level*num_frames), n_points]
            # src_valid_ratios: [N, self.n_levels, 2]
            N, Len_q = sampling_locations.shape[:2]
            sampling_locations = sampling_locations / src_valid_ratios[:, None, None, :, None, :]
            weights_flat = attention_weights.view(N, Len_q, -1)  # [B, Q, n_head * n_level * n_points]
            samples_flat = sampling_locations.view(N, Len_q, -1, 2)  # [B, Q, n_head * n_level * n_points, 2]
            top_weights, top_idx = weights_flat.topk(30, dim=2)  # [B, Q, 30], [B, Q, 30]
            weights_keep = torch.gather(weights_flat, 2, top_idx)  # [B, Q, 30]
            samples_keep = torch.gather(samples_flat, 2, top_idx.unsqueeze(-1).repeat(1, 1, 1, 2))  # [B, Q, 30, 2]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:  # True
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_samples.append(samples_keep)

        if self.return_intermediate:  # true
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
                intermediate_samples)  # output: [4,5,5,256] [4,5,5,2] [4,1,5,30,2]

        return output, reference_points, samples_keep


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def initialize_edges_and_types(batch_size: int, num_frames: int, num_queries_per_frame: int, device: str) -> (
        Tensor, Tensor):
    # 每帧query间有5*4=20条边，总共100条
    # 每帧间对应query有5条边，总共20条
    edge_index = []
    edge_type = []

    for batch in range(batch_size):
        start_idx = batch * num_frames * num_queries_per_frame

        # 每个帧内的双向边
        for frame in range(num_frames):
            frame_start_idx = start_idx + frame * num_queries_per_frame
            for i in range(num_queries_per_frame):
                for j in range(i + 1, num_queries_per_frame):
                    edge_index.append([frame_start_idx + i, frame_start_idx + j])
                    edge_type.append(0)
                    edge_index.append([frame_start_idx + j, frame_start_idx + i])
                    edge_type.append(0)

        # 不同帧间的单向边
        for frame in range(num_frames - 1):
            frame_start_idx = start_idx + frame * num_queries_per_frame
            next_frame_start_idx = frame_start_idx + num_queries_per_frame
            for i in range(num_queries_per_frame):
                edge_index.append([frame_start_idx + i, next_frame_start_idx + i])
                edge_type.append(1)

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    edge_type = (torch.tensor(edge_type, dtype=torch.long, device=device)
                 .view(-1).contiguous())  # ensure edge_type is a flat tensor

    return edge_index, edge_type


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


if __name__ == '__main__':
    edge_index, edge_type = initialize_edges_and_types(1, 1, 5, 'cpu')
    rgcn_conv = RGCNConv(256, 256, 2)
    gat_conve = GATv2Conv(256, 256, 4, concat=False)
    out = rgcn_conv(torch.rand((25, 256)), edge_index, edge_type)
    out = gat_conve(out, edge_index)
    print(out.shape)

    print(edge_index, edge_type)
