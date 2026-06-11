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

from typing import Optional
import torch
from einops import einsum, rearrange, repeat
from torch import nn
# torch.backends.cuda.preferred_linalg_library('magma')

from vggt.heads.utils import cam_quat_xyzw_to_world_quat_wxyz
from vggt.utils.specs import Gaussians
from vggt.utils.geometry import affine_inverse, get_world_rays, sample_image_grid
from vggt.utils.pose_align import batch_align_poses_umeyama
from vggt.utils.sh_helpers import rotate_sh
from vggt.layers import Mlp,MlpFP32


import logging
import numpy as np

class GaussianAdapter(nn.Module):

    def __init__(
        self,
        sh_degree: int = 0,
        pred_color: bool = False,
        pred_offset_depth: bool = False,
        pred_offset_xy: bool = True,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
        filter_with_opacity: float = 1.0,
        gs_options = None,
    ):
        super().__init__()
        # self.sh_degree = sh_degree
        self.sh_degree = gs_options["sh_degree"] if gs_options is not None else sh_degree
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.filter_with_opacity = filter_with_opacity

        # self.pred_offset_depth =False
        # self.pred_offset_xy = False
        self.color_sh = gs_options["color_sh"] if gs_options is not None else False
        self.offset = False

        self.gs_options = gs_options

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        if not pred_color:
            self.register_buffer(
                "sh_mask",
                torch.ones((self.d_sh,), dtype=torch.float32),
                persistent=False,
            )
            for degree in range(1, sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    # def to(self, *args, **kwargs):
    #     self.trunk = self.trunk.to(*args, **kwargs)
    #     self.token_norm = self.token_norm.to(*args, **kwargs)
    #     self.trunk_norm = self.trunk_norm.to(*args, **kwargs)
    #     self.poseLN_modulation = self.poseLN_modulation.to(*args, **kwargs)
    #     self.adaln_norm = self.adaln_norm.to(*args, **kwargs)
    #     self.pose_branch = self.pose_branch.to(*args, **kwargs)

    #     # keep these parameters in FP32
    #     args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
    #     self.empty_pose_tokens = nn.Parameter(self.empty_pose_tokens.to(*args, **kwargs))
    #     self.embed_pose = self.embed_pose.to(*args, **kwargs)

    #     return self

    def forward(
        self,
        depths: torch.Tensor,  # "*#batch"
        opacities: torch.Tensor,  # "*#batch" | "*#batch _"
        raw_gaussians: torch.Tensor,  # "*#batch _"
        image_shape: tuple[int, int],
        extrinsics: torch.Tensor = None,  # "*#batch 4 4"
        intrinsics: torch.Tensor = None,  # "*#batch 3 3"
        eps: float = 1e-8,
        gt_extrinsics: Optional[torch.Tensor] = None,  # "*#batch 4 4"
        sh_RGB = None,
        **kwargs,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        H, W = image_shape
        b, v = raw_gaussians.shape[:2]

        # get cam2worlds and intr_normed to adapt to 3DGS codebase
        if extrinsics is not None:
            cam2worlds = affine_inverse(extrinsics)
            intr_normed = intrinsics.clone().detach()
            intr_normed[..., 0, :] /= W
            intr_normed[..., 1, :] /= H

        # 1. compute 3DGS means
        # 1.1) offset the predicted depth if needed
        # if not self.offset:
        #     logging.warning("注意 手动将 pred_offset_depth 和 pred_offset_xy 设为 False 并且删除了输出的")
        if self.pred_offset_depth:
        # if False:
            if self.offset:
                gs_depths = depths + raw_gaussians[..., -1]
            else:
                gs_depths = depths
            raw_gaussians = raw_gaussians[..., :-1]
        else:
            gs_depths = depths
        # 1.2) align predicted poses with GT if needed
        if gt_extrinsics is not None and not torch.equal(extrinsics, gt_extrinsics):
            try:
                _, _, pose_scales = batch_align_poses_umeyama(
                    gt_extrinsics.detach().float(),
                    extrinsics.detach().float(),
                )
            except Exception:
                pose_scales = torch.ones_like(extrinsics[:, 0, 0, 0])
            pose_scales = torch.clamp(pose_scales, min=1 / 3.0, max=3.0)
            cam2worlds[:, :, :3, 3] = cam2worlds[:, :, :3, 3] * rearrange(
                pose_scales, "b -> b () ()"
            )  # [b, i, j]
            gs_depths = gs_depths * rearrange(pose_scales, "b -> b () () ()")  # [b, v, h, w]



        # 1.3) casting xy in image space
        xy_ray, _ = sample_image_grid((H, W), device)
        xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # b v h w xy
        # offset xy if needed
        if self.pred_offset_xy:
        # if False:
            if self.offset:
                pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
                offset_xy = raw_gaussians[..., :2]
                xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]  # skip the offset_xy
        # # 1.4) unproject depth + xy to world ray
        # origins, directions = get_world_rays(
        #     xy_ray,
        #     repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
        #     repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        # )
        # gs_means_world = origins + directions * gs_depths[..., None]
        # gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")
        # 生成高斯中心均值
            
        def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
            """
            Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
            """
            if ext.shape == (4, 4):
                return ext
            if ext.shape == (3, 4):
                H = np.eye(4, dtype=ext.dtype)
                H[:3, :4] = ext
                return H
            raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones_like(us)
        pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

        pts_all_batch, col_all_batch = [], []
        for ii in range(b):
            pts_all, col_all = [], []
            for i in range(extrinsics.shape[1]):
                d = gs_depths[ii,i].cpu().detach().numpy()  # (H,W)
                valid = np.isfinite(d) & (d > 0)

                d_flat = d.reshape(-1)
                vidx = np.flatnonzero(valid.reshape(-1))

                K_inv = np.linalg.inv(intrinsics[0][i].cpu().detach().numpy())  # (3,3)
                if gt_extrinsics is not None:
                    c2w = np.linalg.inv(_as_homogeneous44(gt_extrinsics[0][i].float().cpu().detach().numpy()))  # (4,4)
                else:
                    c2w = np.linalg.inv(_as_homogeneous44(extrinsics[0][i].cpu().detach().numpy()))  # (4,4)

                rays = K_inv @ pix[vidx].T  # (3,M)
                Xc = rays * d_flat[vidx][None, :]  # (3,M)
                Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
                Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

                pts_all.append(Xw)
            pts_all_flatten = np.concatenate(pts_all, axis=0)
            gs_means_world = torch.from_numpy(pts_all_flatten).to(device).unsqueeze(0)
            pts_all_batch.append(gs_means_world)
        gs_means_world = torch.cat(pts_all_batch, dim=0)  # b (v h w) 3




        # 2. compute other GS attributes


        # try:
        scale_dim = 2 if self.gs_options['gs_mode'] == "2DGS" else 3
        scales, rotations, sh = raw_gaussians.split((scale_dim, 4, 3 * self.d_sh), dim=-1)
        # except:
        #     scales, rotations, sh = raw_gaussians.split((3, 4, 3 * 9), dim=-1)
        #     sh = sh[:, :, :, :, :3 * self.d_sh]
        if (sh_RGB is not None) and (self.color_sh):
            sh = sh_RGB.to(sh.dtype)

        # 2.1) 3DGS scales
        # make the scale invarient to resolution
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        # 基于内参和像素 生成一个尺度因子
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        # 2.2) 3DGS quaternion (world space)
        # due to historical issue, assume quaternion in order xyzw, not wxyz
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        # rotate them to world space
        cam_quat_xyzw = rearrange(rotations, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds,
            "b v i j -> b (v h w) i j",
            h=H,
            w=W,
        )
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        # 生成GS在世界坐标系下的rot
        gs_rotations_world = world_quat_wxyz  # b (v h w) c

        # 2.3) 3DGS color / SH coefficient (world space)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if not self.pred_color:
            sh = sh * self.sh_mask

        if self.pred_color or self.sh_degree == 0:
            # predict pre-computed color or predict only DC band, no need to transform
            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        # 2.4) 3DGS opacity
        gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")
        
        keep_count = gs_opacities.shape[1]
        keep_ratio = float(min(max(self.filter_with_opacity, 0.0), 1.0))
        keep_count = min(keep_count, max(1, int(gs_opacities.shape[1] * keep_ratio)))
        keep_count = min(keep_count, gs_opacities.shape[1])
        # if keep_ratio < 1.0 and gs_opacities.shape[1] > 1:
            # opacity_scores = gs_opacities.mean(dim=-1)
            # topk_idx = torch.topk(gs_opacities, keep_count, dim=1).indices
            # gs_means_world = self._gather_by_indices(gs_means_world, topk_idx)
            # gs_scales = self._gather_by_indices(gs_scales, topk_idx)
            # gs_rotations_world = self._gather_by_indices(gs_rotations_world, topk_idx)
            # gs_sh_world = self._gather_by_indices(gs_sh_world, topk_idx)
            # gs_opacities = self._gather_by_indices(gs_opacities, topk_idx)


        return Gaussians(
            means=gs_means_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities,
            scales=gs_scales,
            rotations=gs_rotations_world,
            # topk_mask = topk_idx
        )

    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        pixel_size: torch.Tensor,  # "*#batch 2"
        multiplier: float = 0.1,
    ) -> torch.Tensor:  # " *batch"
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return 1 if self.pred_color else (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        # provided as reference to the gs_dpt output dim
        raw_gs_dim = 0
        if self.pred_offset_xy:
            raw_gs_dim += 2
        raw_gs_dim += 3  # scales
        raw_gs_dim += 4  # quaternion
        raw_gs_dim += 3 * self.d_sh  # color
        if self.pred_offset_depth:
            raw_gs_dim += 1

        return raw_gs_dim

    def _gather_by_indices(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices is None:
            return tensor
        idx_view = indices.view(indices.shape[0], indices.shape[1], *([1] * (tensor.dim() - 2)))
        idx_expanded = idx_view.expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=idx_expanded)
