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
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn
# torch.backends.cuda.preferred_linalg_library('magma')

from vggt.models.depth_anything_3.model.utils.transform import cam_quat_xyzw_to_world_quat_wxyz
from vggt.models.depth_anything_3.specs import Gaussians
from vggt.models.depth_anything_3.utils.geometry import affine_inverse, get_world_rays, sample_image_grid
from vggt.models.depth_anything_3.utils.pose_align import batch_align_poses_umeyama
from vggt.models.depth_anything_3.utils.sh_helpers import rotate_sh
from vggt.utils.sh_helpers import RGB2SH

import logging

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
        init_sh_dc_from_image: bool = True,
        gs_mode: str = "3DGS",
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.filter_with_opacity = filter_with_opacity
        self.init_sh_dc_from_image = init_sh_dc_from_image
        self.gs_mode = gs_mode
        self.scale_dim = 2 if str(gs_mode).upper() == "2DGS" else 3

        # self.pred_offset_depth =False
        # self.pred_offset_xy = False
        
        # self.offset = False


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

    def forward(
        self,
        extrinsics: torch.Tensor,  # "*#batch 4 4"
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        depths: torch.Tensor,  # "*#batch"
        opacities: torch.Tensor,  # "*#batch" | "*#batch _"
        raw_gaussians: torch.Tensor,  # "*#batch _"
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        gt_extrinsics: Optional[torch.Tensor] = None,  # "*#batch 4 4"
        image_rgb: Optional[torch.Tensor] = None,  # "*#batch 3 h w", expected in [0, 1]
        **kwargs,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        H, W = image_shape
        b, v = raw_gaussians.shape[:2]

        # get cam2worlds and intr_normed to adapt to 3DGS codebase
        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        # 1. compute 3DGS means
        # 1.1) offset the predicted depth if needed
        # if not self.offset:
        #     logging.warning("注意 手动将 pred_offset_depth 和 pred_offset_xy 设为 False 并且删除了输出的")
        
        # if False:
            # if self.offset:
        if self.pred_offset_depth:
            gs_depths = depths + raw_gaussians[..., -1]
        else:
            gs_depths = depths
        raw_gaussians = raw_gaussians[..., :-1]
        # else:
        #     gs_depths = depths
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
            # if self.offset:
            pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
            offset_xy = raw_gaussians[..., :2]
            xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]
        else:
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
            
        # def _as_homogeneous44(ext: torch.Tensor) -> torch.Tensor:
        #     """
        #     Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
        #     """
        #     if tuple(ext.shape) == (4, 4):
        #         return ext
        #     if tuple(ext.shape) == (3, 4):
        #         H44 = torch.eye(4, dtype=ext.dtype, device=ext.device)
        #         H44[:3, :4] = ext
        #         return H44
        #     raise ValueError(f"extrinsic must be (4,4) or (3,4), got {tuple(ext.shape)}")

        # y_coords, x_coords = torch.meshgrid(
        #     torch.arange(H, device=device, dtype=torch.float32),
        #     torch.arange(W, device=device, dtype=torch.float32),
        #     indexing="ij",
        # )
        # ones = torch.ones_like(x_coords)
        # pix = torch.stack([x_coords, y_coords, ones], dim=-1).reshape(-1, 3)  # (H*W, 3)

        # extr_for_points = gt_extrinsics if gt_extrinsics is not None else extrinsics
        # means_per_batch = []
        # for batch_idx in range(b):
        #     pts_all = []
        #     for view_idx in range(v):
        #         d = gs_depths[batch_idx, view_idx].float().reshape(-1)
        #         valid = torch.isfinite(d) & (d > 0)
        #         d = torch.where(valid, d, torch.zeros_like(d))

        #         K_inv = torch.linalg.inv(intrinsics[batch_idx, view_idx].float())
        #         c2w = torch.linalg.inv(_as_homogeneous44(extr_for_points[batch_idx, view_idx].float()))

        #         rays = K_inv @ pix.t()  # (3, H*W)
        #         Xc = rays * d.unsqueeze(0)  # (3, H*W)
        #         Xc_h = torch.cat([Xc, torch.ones((1, Xc.shape[1]), device=device, dtype=Xc.dtype)], dim=0)
        #         Xw = (c2w @ Xc_h)[:3].t()  # (H*W, 3)

        #         pts_all.append(Xw)

        #     means_per_batch.append(torch.cat(pts_all, dim=0))

        # gs_means_world = torch.stack(means_per_batch, dim=0).to(dtype=dtype)
        origins, directions = get_world_rays(
            xy_ray,
            repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
            repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        )
        gs_means_world = origins + directions * gs_depths[..., None]
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")




        # 2. compute other GS attributes
        try:
            scales, rotations, sh = raw_gaussians.split((self.scale_dim, 4, 3 * self.d_sh), dim=-1)
        except:
            scales, rotations, sh = raw_gaussians.split((self.scale_dim, 4, 3 * 9), dim=-1)
            sh = sh[:, :, :, :, :3 * self.d_sh]

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

        if self.init_sh_dc_from_image and image_rgb is not None:
            if image_rgb.ndim == 4:
                image_rgb = image_rgb.unsqueeze(0)

            if image_rgb.shape[-2:] != (H, W):
                image_rgb = F.interpolate(
                    rearrange(image_rgb, "b v c h w -> (b v) c h w"),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                image_rgb = rearrange(image_rgb, "(b v) c h w -> b v c h w", b=b, v=v)

            image_rgb = image_rgb.to(dtype=sh.dtype, device=sh.device)
            img_min = image_rgb.amin().item()
            img_max = image_rgb.amax().item()
            if img_min < -0.1 or img_max > 1.1:
                imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=sh.device, dtype=sh.dtype).view(1, 1, 3, 1, 1)
                imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=sh.device, dtype=sh.dtype).view(1, 1, 3, 1, 1)
                image_rgb = image_rgb * imagenet_std + imagenet_mean

            image_rgb = image_rgb.clamp(0.0, 1.0)
            sh_dc_init = RGB2SH(rearrange(image_rgb, "b v c h w -> b v h w c")).unsqueeze(-1)
            sh_dc_residual = sh[..., :1]
            sh_dc_residual = sh_dc_residual - sh_dc_residual.mean(dim=(2, 3), keepdim=True)
            sh_dc = sh_dc_init + sh_dc_residual
            # if sh.shape[-1] > 1:
            sh = torch.cat([sh_dc, sh[..., 1:]], dim=-1)
            # else:
            #     sh = sh_dc

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
        if keep_ratio < 1.0 and gs_opacities.shape[1] > 1:
            # opacity_scores = gs_opacities.mean(dim=-1)
            topk_idx = torch.topk(gs_opacities, keep_count, dim=1).indices
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
        raw_gs_dim += self.scale_dim  # scales
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
