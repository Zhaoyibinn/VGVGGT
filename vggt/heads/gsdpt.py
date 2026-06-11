# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict as TyDict
from typing import List, Sequence
import torch
import torch.nn as nn

from vggt.heads.dpt_head import DPTHead
from vggt.heads.utils import activate_head_gs, custom_interpolate

from vggt.layers import MlpFP32


class GSDPT(DPTHead):

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        # output_dim: int = 4,
        activation: str = "linear",
        conf_activation: str = "sigmoid",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        conf_dim: int = 1,
        norm_type: str = "idt",  # use to match legacy GS-DPT head, "idt" / "layer"
        fusion_block_inplace: bool = False,
        gs_options = None,
    ) -> None:
        
        scale_dim = 2 if gs_options['gs_mode'] == "2DGS" else 3
        rot_dim = 4
        xyz_dim = 3
        opacity_dim = 1
        output_dim=rot_dim + xyz_dim + scale_dim + opacity_dim
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            # head_name="raw_gs",
            # use_sky_head=False,
            # norm_type=norm_type,
            # fusion_block_inplace=fusion_block_inplace,
        )
        self.conf_dim = conf_dim
        if conf_dim and conf_dim > 1:
            assert (
                conf_activation == "linear"
            ), "use linear prediction when using view-dependent opacity"

        merger_out_dim = features if feature_only else features // 2
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, merger_out_dim // 4, 3, 1, 1),  # fewer channels first
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 4, merger_out_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 2, merger_out_dim, 3, 1, 1),
            nn.GELU(),
        )
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.images_merger.to(*args, **kwargs)
        
        

        # super().to('cuda')
        # self.images_merger.to('cuda')
        
        return self

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> TyDict[str, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []

        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[take_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], ph, pw))

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        out = self.scratch_forward(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        out = custom_interpolate(out, (h_out, w_out), mode="bilinear", align_corners=True)
        image_feats = images.reshape(B * S, 3, H, W)
        merged = self.images_merger(image_feats)
        out = out + merged

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        feat = out
        if feat.dtype != torch.float32:
            main_logits = self.scratch.output_conv2(feat.float().contiguous())
        else:
            main_logits = self.scratch.output_conv2(feat)
        pts3d, conf = activate_head_gs(
            main_logits,
            activation=self.activation,
            conf_activation=self.conf_activation,
            conf_dim=self.conf_dim,
        )

        pts3d = pts3d.view(B, S, *pts3d.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])

        return pts3d,  conf
