from __future__ import annotations

import torch
import torch.nn as nn

from vggt.models.depth_anything_3.model.dpt import DPT
from vggt.models.depth_anything_3.model.utils.head_utils import custom_interpolate


class BackendDenseDPT(DPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scratch.output_conv2 = nn.Identity()
        # 这个主要是因为不参与forward，就放个没用的了
        

    def forward(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        B, S, N, C = feats[0][0].shape
        flat_feats = [feat[0].reshape(B * S, N, C) for feat in feats]

        if chunk_size is None or chunk_size >= (B * S):
            dense = self._forward_impl(flat_feats, H, W, patch_start_idx)
        else:
            dense = torch.cat(
                [
                    self._forward_impl([feat[s0:s1] for feat in flat_feats], H, W, patch_start_idx)
                    for s0 in range(0, B * S, chunk_size)
                    for s1 in [min(s0 + chunk_size, B * S)]
                ],
                dim=0,
            )

        dense = dense.permute(0, 2, 3, 1).contiguous()
        return dense.view(B, S, dense.shape[1], dense.shape[2], dense.shape[3])

    def _forward_impl(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> torch.Tensor:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        fused = self._fuse(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)
        return fused