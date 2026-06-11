
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp,MlpFP32
from vggt.heads.dpt_head import DPTHead
import numpy as np  
import copy
from vggt.utils import transforms
from vggt.layers.tsdf_muti_attention import Attention


class TSDFMapper(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        tsdf_sample_xyz_num = None,
    ) -> None:
        super().__init__()
        self.tsdf_sample_xyz_num = tsdf_sample_xyz_num
        self.tsdf_attention = Attention(
            dim_in,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.tsdf_xyz_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, dim_in),
            nn.LayerNorm(dim_in),
        )
        self.tsdf_out = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.Linear(1024, 1)
        )
        self._tsdf_token = None

    def set_context(self, tsdf_token: torch.Tensor) -> None:
        self._tsdf_token = tsdf_token

    def forward(self, tsdf_xyz: torch.Tensor) -> torch.Tensor:
        if self._tsdf_token is None:
            raise ValueError("TSDFMapper context not set. Call set_context(tsdf_token) before forward().")
        
        tsdf_token = self._tsdf_token.to(tsdf_xyz.dtype)
        if tsdf_xyz.dim() == 5:
            bsz, gx, gy, gz, _ = tsdf_xyz.shape
            tsdf_xyz = tsdf_xyz.reshape(bsz, -1, 3)
        elif tsdf_xyz.dim() == 3:
            bsz = tsdf_xyz.shape[0]
        else:
            raise ValueError(f"Unexpected tsdf_xyz shape: {tsdf_xyz.shape}")

        if tsdf_token.shape[0] != bsz:
            raise ValueError(
                f"Batch size mismatch: tsdf_xyz batch {bsz} vs tsdf_token batch {tsdf_token.shape[0]}"
            )

        tsdf_xyz = tsdf_xyz.to(tsdf_token.device).float()
        tsdf_token_batch = tsdf_token.reshape(tsdf_token.shape[0], -1, tsdf_token.shape[-1])

        assert tsdf_xyz.shape[1] >= self.tsdf_sample_xyz_num, "GT tsdf 采样点太少了 比设定的batch还少"
        # if (
        #     self.tsdf_sample_xyz_num is None
        #     or self.tsdf_sample_xyz_num <= 0
        #     or tsdf_xyz.shape[1] <= self.tsdf_sample_xyz_num
        # ):
        #     tsdf_xyz_tokens = self.tsdf_xyz_encoder(tsdf_xyz)
        #     tokens_result = self.tsdf_attention(tsdf_xyz_tokens, tsdf_token_batch, pos=None)
        #     tsdf_pred = self.tsdf_out(tokens_result).squeeze(-1)
        #     return tsdf_pred

        tsdf_pred_chunks = []
        for start in range(0, tsdf_xyz.shape[1], self.tsdf_sample_xyz_num):
            end = start + self.tsdf_sample_xyz_num
            tsdf_xyz_chunk = tsdf_xyz[:, start:end, :]
            tsdf_xyz_tokens = self.tsdf_xyz_encoder(tsdf_xyz_chunk)
            tokens_result = self.tsdf_attention(tsdf_xyz_tokens, tsdf_token_batch, pos=None)
            tsdf_pred_chunks.append(self.tsdf_out(tokens_result).squeeze(-1))

        tsdf_pred = torch.cat(tsdf_pred_chunks, dim=1)
        return tsdf_pred


class VSDFHead(nn.Module):
    """
    VSDFHead
    """

    def __init__(
        self,
        dim_in,
        vsdf_options,
        ):
        super().__init__()

        self.voxel_size = vsdf_options['voxel_size']
        self.N_VOX = vsdf_options['N_VOX']
        self.tsdf_sample_xyz_num = vsdf_options['tsdf_sample_xyz_num']
        self.tsdf_mapper = TSDFMapper(
            dim_in=dim_in,
            num_heads=16,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            qk_norm=False,
            fused_attn=True,
            rope=None,
            tsdf_sample_xyz_num = self.tsdf_sample_xyz_num
        )

        # self.dpt_feature_layer = DPTHead(
        #     dim_in=dim_in,
        #     output_dim = 16
        # )

    # def to(self, *args, **kwargs):
    #     pass
    def to(self, *args, **kwargs):


        # keep these parameters in FP32
        args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
        self.tsdf_mapper = self.tsdf_mapper.to(*args, **kwargs)

        return self
    
    def forward(self, predictions, aggregated_tokens_list,images, patch_start_idx,gt_data = None):
        # world_points = predictions['world_points']
        # world_points_conf = predictions['world_points_conf']
        # B,N,H,W,C = world_points.shape
        tsdf_token = aggregated_tokens_list[23]
        self.tsdf_mapper.set_context(tsdf_token)
        return tsdf_token, self.tsdf_mapper


    # def forward_old(self, predictions, aggregated_tokens_list,images, patch_start_idx,gt_data = None):
    #     world_points = predictions['world_points']
    #     world_points_conf = predictions['world_points_conf']
    #     B,N,H,W,C = world_points.shape

    #     dpt_feature_whole = self.dpt_feature_layer(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
    #     dpt_feature, dpt_feature_conf = dpt_feature_whole[0],dpt_feature_whole[1]
    #     transform = []
    #     random_rotation = False
    #     random_translation = False
    #     paddingXY = .1
    #     paddingZ = .025
    #     transform += [
    #                 transforms.RandomTransformSpace(
    #                     self.N_VOX, self.voxel_size, random_rotation, random_translation,
    #                     paddingXY, paddingZ, max_epoch=10),
    #                 transforms.IntrinsicsPoseToProjection(N, 4),
    #                 ]
    #     transforms = transforms.Compose(transform)
    #     tsdf_gt_device = torch.device('cpu')
    #     for i in range(B):
    #         depth = predictions['depth'][i]
    #         estr = predictions['extrinsics'][i]
    #         intr = predictions['intrinsics'][i]

    #         if gt_data!=None:
    #             gt_estr = gt_data['extrinsics'][i]
    #             gt_intr = gt_data['intrinsics'][i]
    #             gt_voxel_xyz = gt_data['voxel_xyz'][i]
    #             gt_tsdf = gt_data['tsdf'][i]
    #             gt_images= gt_data['images'][i]
    #             gt_depths = gt_data['depths'][i]

    #         else:
    #             gt_estr = estr
    #             gt_intr = intr

    #         # points_world = self.depth_to_world(depth, intr, estr, farthest_percent=0.05)


    #         pad_row = torch.zeros(*gt_estr.shape[:-2], 1, 4, device=gt_estr.device, dtype=gt_estr.dtype)
    #         pad_row[..., 0, 3] = 1.

    #         imgs_for_trans = gt_images.to(tsdf_gt_device)
    #         vol_origin_for_trans = gt_voxel_xyz[0].to(tsdf_gt_device)
    #         depths_for_trans = gt_depths.to(tsdf_gt_device)
    #         tsdf_full_list_for_trans = [gt_tsdf.to(tsdf_gt_device)]
    #         estr_for_trans = torch.cat([gt_estr, pad_row], dim=-2).to(tsdf_gt_device)
    #         intr_for_trans = gt_intr.to(tsdf_gt_device)

    #         items_for_trans = {
    #                         'imgs': imgs_for_trans,
    #                         'depth': depths_for_trans,
    #                         'intrinsics': intr_for_trans,
    #                         'extrinsics': estr_for_trans,
    #                         'tsdf_list_full': tsdf_full_list_for_trans,
    #                         'vol_origin': vol_origin_for_trans,
    #                         'scene': "test",
    #                         'fragment': "test",
    #                         'epoch': [0],
    #                         }
    #         transforms(items_for_trans)

    #         def generate_grid(n_vox, interval):
    #             with torch.no_grad():
    #                 # Create voxel grid
    #                 grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
    #                 grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
    #                 grid = grid.unsqueeze(0).cuda().float()  # 1 3 dx dy dz
    #                 grid = grid.view(1, 3, -1)
    #             return grid
    #         coords = generate_grid(self.N_VOX, 1)[0]
    #         dpt_feature_B = dpt_feature[i].unsqueeze(1).permute(0,1,4,2,3)
    #         up_coords = []
    #         up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * i, coords]))
    #         up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()

    #         # 每一个点对应的体素索引

    #     print("VSDFHead forward called")


    @staticmethod
    def depth_to_world(depth, intr, extr, farthest_percent=0.):
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.dim() == 3:
            all_points = []
            num_views = depth.shape[0]
            intr_per_view = intr if intr.dim() == 3 else intr.unsqueeze(0).expand(num_views, 3, 3)
            extr_per_view = extr if extr.dim() == 3 else extr.unsqueeze(0).expand(num_views, 3, 4)
            for v in range(num_views):
                pts = VSDFHead._depth_to_world_single(depth[v], intr_per_view[v], extr_per_view[v], farthest_percent)
                if pts.numel():
                    all_points.append(pts)
            return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 3), device=depth.device)
        return VSDFHead._depth_to_world_single(depth, intr, extr, farthest_percent)

    @staticmethod
    def _depth_to_world_single(depth, intr, extr, farthest_percent=0.):
        device = depth.device
        valid_flat = depth.reshape(-1) > 0
        if not torch.any(valid_flat):
            return torch.empty((0, 3), device=device)

        im_h, im_w = depth.shape
        ys = torch.arange(im_h, device=device).unsqueeze(1).expand(im_h, im_w)
        xs = torch.arange(im_w, device=device).unsqueeze(0).expand(im_h, im_w)
        ys_flat = ys.reshape(-1)[valid_flat].float()
        xs_flat = xs.reshape(-1)[valid_flat].float()
        z = depth.reshape(-1)[valid_flat]

        if farthest_percent > 0 and farthest_percent < 1:
            threshold = torch.quantile(z, 1 - farthest_percent)
            mask = z <= threshold
            if not torch.any(mask):
                return torch.empty((0, 3), device=device)
            xs_flat = xs_flat[mask]
            ys_flat = ys_flat[mask]
            z = z[mask]
        elif farthest_percent >= 1:
            return torch.empty((0, 3), device=device)

        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        x = (xs_flat - cx) * z / fx
        y = (ys_flat - cy) * z / fy

        points_cam = torch.stack([x, y, z], dim=1)
        ones = torch.ones(points_cam.shape[0], 1, device=device)
        cam_h = torch.cat([points_cam, ones], dim=1)
        extr4 = extr
        if extr.shape[0] == 3:
            pad_row = torch.tensor([[0, 0, 0, 1.]], dtype=extr.dtype, device=device)
            extr4 = torch.cat([extr, pad_row], dim=0)
        world = (torch.inverse(extr4) @ cam_h.T).T[:, :3]
        return world
