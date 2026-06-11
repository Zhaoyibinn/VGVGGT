# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor

from fused_ssim import fused_ssim
from tqdm import tqdm
import numpy as np
import copy
import logging
from typing import Any
from vggt.models.GS.utils.build_camera import build_gs_camera
from vggt.models.depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, unproject_depth
from vggt.models.GS.utils.point_utils import depth_to_normal

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'vggt', 'models', 'amb3r', 'thirdparty'))
from moge.moge.train.losses import scale_invariant_alignment

import lpips


def visualize_tsdf_slices(
    tsdf: torch.Tensor,
    out_dir: str,
    prefix: str = "tsdf",
    num_slices: int = 8,
    axis: str = "z",
    vmax: float | None = None,
):
    """
    Save 2D slice visualizations for TSDF volumes.

    Args:
        tsdf: Tensor with shape (B, X, Y, Z) or (B, X, Y, Z, 1)
        out_dir: Output directory for slice images
        prefix: Filename prefix for saved images
        num_slices: Number of slices to save per sample
        axis: Slice axis, one of "x", "y", "z"
        vmax: Fixed max absolute value for color scaling (auto if None)
    """
    import matplotlib.pyplot as plt

    if tsdf.dim() == 5 and tsdf.shape[-1] == 1:
        tsdf = tsdf.squeeze(-1)
    if tsdf.dim() != 4:
        raise ValueError(f"Expected tsdf with shape (B, X, Y, Z) or (B, X, Y, Z, 1), got {tsdf.shape}")

    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"axis must be one of 'x', 'y', 'z', got {axis}")

    os.makedirs(out_dir, exist_ok=True)

    axis_map = {"x": 1, "y": 2, "z": 3}
    axis_dim = axis_map[axis]
    total_slices = tsdf.shape[axis_dim]
    num_slices = max(1, min(int(num_slices), total_slices))
    indices = torch.linspace(0, total_slices - 1, steps=num_slices).round().long().tolist()

    for b in range(tsdf.shape[0]):
        for idx in indices:
            if axis == "x":
                slice_2d = tsdf[b, idx, :, :]
            elif axis == "y":
                slice_2d = tsdf[b, :, idx, :]
            else:
                slice_2d = tsdf[b, :, :, idx]

            vmax_val = vmax
            if vmax_val is None:
                vmax_val = float(slice_2d.abs().max().item()) if slice_2d.numel() > 0 else 1.0
                if vmax_val == 0:
                    vmax_val = 1.0

            plt.figure(figsize=(4, 4))
            im = plt.imshow(slice_2d.numpy(), cmap="seismic", vmin=-vmax_val, vmax=vmax_val)
            plt.colorbar(im)
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{prefix}_b{b}_{axis}{idx:04d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        
@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, gs=None, tsdf=None, align_moge=False, refinement=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.gs = gs
        self.track = track
        self.tsdf = tsdf
        self.align_moge = align_moge
        self.refinement = refinement or {"enabled": False}
        self.lpips_model = lpips.LPIPS(net='vgg').cuda()
        for param in self.lpips_model.parameters():
            param.requires_grad_(False)


    def forward(self, predictions, batch,train = False) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        if self._should_use_refinement(predictions):
            return self._forward_refinement(predictions, batch, train=train)
        return self._forward_single(predictions, batch, train=train)

    def _should_use_refinement(self, predictions: dict[str, Any]) -> bool:
        return bool(self.refinement.get("enabled", False) and predictions.get("stages", None) and len(predictions["stages"]) >= 2)

    @staticmethod
    def _normalize_prediction_geometry(predictions: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        if not all(key in predictions for key in ("extrinsics", "intrinsics")):
            return predictions

        with torch.amp.autocast("cuda", enabled=False):
            intrinsics = predictions["intrinsics"].float()
            extrinsics_h = as_homogeneous(predictions["extrinsics"].float())

            first_cam_inv = affine_inverse(extrinsics_h[:, 0])
            normalized_extrinsics_h = torch.matmul(extrinsics_h, first_cam_inv.unsqueeze(1))

            normalized_extrinsics = normalized_extrinsics_h[:, :, :3, :]
            image_hw = batch["images"].shape[-2:]
            pose_enc = extri_intri_to_pose_encoding(normalized_extrinsics, intrinsics, image_hw)

            predictions["extrinsics"] = normalized_extrinsics
            predictions["intrinsics"] = intrinsics
            predictions["pose_enc"] = pose_enc
            predictions["pose_enc_list"] = [pose_enc]

        return predictions

    @staticmethod
    def _build_global_points(
        predictions: dict[str, Any],
        depth_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if "world_points" in predictions:
            world_points = predictions["world_points"]
            if depth_scale is not None:
                world_points = world_points * depth_scale[..., None, None, None]
            world_points_conf = predictions.get(
                "world_points_conf",
                predictions.get("depth_conf", torch.ones_like(world_points[..., 0])),
            )
            return world_points, world_points_conf
        if not all(key in predictions for key in ("depth", "extrinsics", "intrinsics")):
            return None, None

        with torch.amp.autocast("cuda", enabled=False):
            depth = predictions["depth"].float()
            if depth.ndim == 4:
                depth = depth[..., None]
            if depth_scale is not None:
                depth = depth * depth_scale[..., None, None, None]
            intrinsics = predictions["intrinsics"].float()
            extrinsics = predictions["extrinsics"].float()
            if depth_scale is not None:
                scaled_translation = extrinsics[..., :3, 3:4] * depth_scale[..., None, None]
                extrinsics = torch.cat((extrinsics[..., :3, :3], scaled_translation), dim=-1)
            extrinsics_h = as_homogeneous(extrinsics)
            c2w = affine_inverse(extrinsics_h)
            world_points = unproject_depth(depth, intrinsics, c2w=c2w)
            world_points_conf = predictions.get("depth_conf", torch.ones_like(depth[..., 0]))
        return world_points, world_points_conf

    def _forward_single(self, predictions, batch, train: bool = False):
        loss_dict, total_loss, _ = self._compute_stage_losses(
            predictions,
            batch,
            train=train,
            prefix="",
            include_gs=True,
            include_tsdf=True,
        )
        loss_dict["objective"] = total_loss
        return loss_dict

    def _forward_refinement(self, predictions, batch, train: bool = False):
        baseline_pred = predictions["stages"][0]
        refined_pred = predictions["stages"][-1]

        baseline_loss_dict, baseline_total, baseline_tasks = self._compute_stage_losses(
            baseline_pred,
            batch,
            train=train,
            prefix="baseline_",
            include_gs=False,
            include_tsdf=False,
        )
        refined_loss_dict, refined_total, refined_tasks = self._compute_stage_losses(
            refined_pred,
            batch,
            train=train,
            prefix="refined_",
            include_gs=True,
            include_tsdf=True,
        )

        loss_dict = {}
        loss_dict.update(baseline_loss_dict)
        loss_dict.update(refined_loss_dict)
        for key, value in refined_loss_dict.items():
            if key.startswith("refined_"):
                loss_dict[key[len("refined_"):]] = value

        improvement_weight = float(self.refinement.get("improvement_weight", 0.0))
        margin = float(self.refinement.get("margin", 0.0))
        penalty_total = refined_total.new_tensor(0.0)
        penalty_usage = {
            "depth": bool(self.refinement.get("use_depth", True)),
            "camera": bool(self.refinement.get("use_camera", True)),
            "point": bool(self.refinement.get("use_point", True)),
        }
        for task_name, enabled in penalty_usage.items():
            if not enabled or task_name not in baseline_tasks or task_name not in refined_tasks:
                continue
            penalty = torch.relu(refined_tasks[task_name] - baseline_tasks[task_name] + margin)
            loss_dict[f"loss_refine_penalty_{task_name}"] = penalty
            penalty_total = penalty_total + penalty

        loss_dict["loss_refinement"] = penalty_total * improvement_weight
        loss_dict["baseline_objective"] = baseline_total
        loss_dict["refined_objective"] = refined_total
        loss_dict["objective"] = refined_total + loss_dict["loss_refinement"]
        return loss_dict

    def _compute_stage_losses(
        self,
        predictions,
        batch,
        train: bool,
        prefix: str,
        include_gs: bool,
        include_tsdf: bool,
    ):
        total_loss = 0
        loss_dict = {}
        task_totals = {}
        depth_scale = None
        predictions = self._normalize_prediction_geometry(predictions, batch)

        if self.depth is not None and "depth" in predictions:
            depth_loss_dict, depth_scale = compute_depth_loss(predictions, batch, align_scale=self.align_moge, **self.depth)
            depth_total = (depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]) * self.depth["weight"]
            total_loss = total_loss + depth_total
            task_totals["depth"] = depth_total
            for key, value in depth_loss_dict.items():
                loss_dict[f"{prefix}{key}"] = value

        if self.camera is not None and "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, depth_scale=depth_scale, **self.camera)
            camera_total = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_total
            task_totals["camera"] = camera_total
            for key, value in camera_loss_dict.items():
                loss_dict[f"{prefix}{key}"] = value

        if self.point is not None:
            world_points, world_points_conf = self._build_global_points(predictions, depth_scale=depth_scale)
            if world_points is not None:
                point_predictions = dict(predictions)
                point_predictions["world_points"] = world_points
                point_predictions["world_points_conf"] = world_points_conf
                point_loss_dict = compute_point_loss(point_predictions, batch, **self.point)
            else:
                point_loss_dict = None
        else:
            point_loss_dict = None

        if point_loss_dict is not None:
            point_total = (point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]) * self.point["weight"]
            total_loss = total_loss + point_total
            task_totals["point"] = point_total
            for key, value in point_loss_dict.items():
                loss_dict[f"{prefix}{key}"] = value

        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")

        if include_gs and self.gs is not None and 'GS_render_pkgs' in predictions:
            gs_loss_dict = self.compute_gs_loss(predictions, batch, depth_scale=depth_scale, train=train)
            for key, value in gs_loss_dict.items():
                loss_dict[f"{prefix}{key}"] = value
            loss_gs = (
                gs_loss_dict["loss_gs_rgb"]
                + gs_loss_dict["loss_gs_lpips"]
                + gs_loss_dict["loss_gs_normal"]
                + gs_loss_dict["loss_gs_dist"]
                + gs_loss_dict["loss_gs_depth"]
                + gs_loss_dict["depth_align_loss"]
            )
            total_loss = total_loss + loss_gs * self.gs["weight"]

        if include_tsdf and self.tsdf is not None and 'tsdf_mapper' in predictions:
            with torch.amp.autocast('cuda',enabled=False):
                tsdf_loss_dict = self.compute_sdf_loss(predictions, batch)
            for key, value in tsdf_loss_dict.items():
                loss_dict[f"{prefix}{key}"] = value
            total_loss = total_loss + (tsdf_loss_dict["loss_tsdf_conf"] + tsdf_loss_dict["loss_tsdf"]) * self.tsdf.get("weight", 1.0)
            if not train:
                with torch.amp.autocast('cuda',enabled=False):
                    with torch.no_grad():
                        sdf_worldpcd_loss_dict = self.sdf_worldpcd_loss(predictions, batch)
                        for key, value in sdf_worldpcd_loss_dict.items():
                            loss_dict[f"{prefix}{key}"] = value

        loss_dict[f"{prefix}objective"] = total_loss
        return loss_dict, total_loss, task_totals

    def sdf_worldpcd_loss(self,predictions, batch, **kwargs):
        tsdf_mapper = predictions['tsdf_mapper']
        tsdf_xyz = batch['voxel_xyz']
        tsdf = batch['tsdf']
        tsdf = tsdf.to(tsdf_xyz.device).float()

        world_points, _ = self._build_global_points(predictions)
        if world_points is None:
            raise KeyError("sdf_worldpcd_loss requires depth, intrinsics, and extrinsics to reconstruct world points")

        world_pcd = world_points

        B = tsdf_xyz.shape[0]

        world_pcd = world_pcd.reshape(B, -1, 3)




        # import open3d as o3d
        # import numpy as np 

        # less_beishu = 5
        # tsdf_xyz_less = tsdf_xyz[:,::less_beishu,::less_beishu,::less_beishu]
        # tsdf_xyz_less_flatten = tsdf_xyz_less.reshape(B, -1, 3)
        # tsdf = tsdf.reshape(B, -1)

        # voxel_pts_world = tsdf_xyz.cpu().detach().numpy()[0]
        # tsdf_vol = tsdf.cpu().detach().numpy()[0]
        # # Ensure voxel_pts_world matches tsdf_vol grid shape
        # if voxel_pts_world.ndim == 2 and voxel_pts_world.shape[1] == 3:
        #     voxel_pts_world = voxel_pts_world.reshape(*tsdf_vol.shape, 3)
        # else:
        #     assert voxel_pts_world.shape[:3] == tsdf_vol.shape, (
        #         f"voxel_pts_world shape {voxel_pts_world.shape} mismatches tsdf_vol {tsdf_vol.shape}")

        # # mask voxels with small absolute tsdf
        # mask = np.abs(tsdf_vol) < 0.02
        # points = voxel_pts_world[mask]

        # if points.size == 0:
        #     print("No voxels satisfy |tsdf| < 0.02")
        # else:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        #     save_path = "test_sdf.ply"
        #     o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
        #     print(f"Saved {len(points)} points to {save_path}")
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(world_pcd.cpu().detach().numpy()[0].astype(np.float64)[::10])
        # o3d.io.write_point_cloud("test.ply", pcd, write_ascii=True)
        


        # 可视化预测的tsdf
        # less_beishu = 2
        # tsdf_xyz_less = tsdf_xyz[:,::less_beishu,::less_beishu,::less_beishu]
        # tsdf_xyz_less_flatten = tsdf_xyz_less.reshape(B, -1, 3)
        
        # visualize_tsdf_slices(
        #     tsdf.detach().cpu(),
        #     out_dir="logs/tsdf_slices_gt",
        #     prefix="tsdf",
        #     num_slices=8,
        #     axis="z",
        #     vmax=None,
        # )
        # pcd_batch_size = 99999999999 # 在model里面设置了batch采样，所以这里就可以全都一起丢进去
        # tsdf_pred = []
        # for world_pcd_batch in torch.split(tsdf_xyz_less_flatten, pcd_batch_size, dim=1):
        #     tsdf_pred_batch = tsdf_mapper(world_pcd_batch)  # (B, N)
        #     tsdf_pred_batch = check_and_fix_inf_nan(tsdf_pred_batch, "tsdf_pred", hard_max=None)
        #     tsdf_pred.append(tsdf_pred_batch.cpu().detach().numpy())
        #     del tsdf_pred_batch
        #     torch.cuda.empty_cache()
        # tsdf_pred = np.concatenate(tsdf_pred, axis=1)
        # tsdf_pred_torch_3d = torch.tensor(tsdf_pred).reshape(tsdf_xyz_less.shape[:-1])
        # visualize_tsdf_slices(
        #     tsdf_pred_torch_3d.detach().cpu(),
        #     out_dir="logs/tsdf_slices",
        #     prefix="tsdf",
        #     num_slices=8,
        #     axis="z",
        #     vmax=None,
        # )



        pcd_batch_size = 99999999999 # 在model里面设置了batch采样，所以这里就可以全都一起丢进去
        tsdf_pred = []
        for world_pcd_batch in torch.split(world_pcd, pcd_batch_size, dim=1):
            tsdf_pred_batch, _ = tsdf_mapper(world_pcd_batch)  # (B, N)
            tsdf_pred_batch = check_and_fix_inf_nan(tsdf_pred_batch, "tsdf_pred", hard_max=None)
            tsdf_pred.append(tsdf_pred_batch.cpu().detach().numpy())
            del tsdf_pred_batch
            torch.cuda.empty_cache()
        tsdf_pred = np.concatenate(tsdf_pred, axis=1)

        sdf_worldpcd_error = np.abs(tsdf_pred).mean()
        return {"sdf_worldpcd_error": sdf_worldpcd_error}




        

    def compute_sdf_loss(self,predictions, batch, **kwargs):
        tsdf_mapper = predictions['tsdf_mapper']
        tsdf_xyz = batch['voxel_xyz']
        tsdf = batch['tsdf']
        

        B = tsdf_xyz.shape[0]

        # Lazy init projection heads so their output dim matches tsdf_token C
        # if (self.tsdf_xyz_encoder is None) or (self.tsdf_xyz_encoder.out_features != C):

        

        # Encode XYZ -> tokens
        tsdf = tsdf.to(tsdf_xyz.device).float()

        # import numpy as np
        # import open3d as o3d

        # voxel_pts_world = tsdf_xyz.cpu().detach().numpy()[0]
        # tsdf_vol = tsdf.cpu().detach().numpy()[0]
        # # Ensure voxel_pts_world matches tsdf_vol grid shape
        # if voxel_pts_world.ndim == 2 and voxel_pts_world.shape[1] == 3:
        #     voxel_pts_world = voxel_pts_world.reshape(*tsdf_vol.shape, 3)
        # else:
        #     assert voxel_pts_world.shape[:3] == tsdf_vol.shape, (
        #         f"voxel_pts_world shape {voxel_pts_world.shape} mismatches tsdf_vol {tsdf_vol.shape}")

        # # mask voxels with small absolute tsdf
        # mask = np.abs(tsdf_vol) < 0.02
        # points = voxel_pts_world[mask]

        # if points.size == 0:
        #     print("No voxels satisfy |tsdf| < 0.02")
        # else:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        #     save_path = "test.ply"
        #     o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
        #     print(f"Saved {len(points)} points to {save_path}")
        # pcd_vggt = predictions['world_points'][0].reshape([-1,3]).cpu().detach().numpy()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_vggt.astype(np.float64))
        # save_path = "test.ply"
        # o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)


        tsdf_train  = predictions.get("tsdf_train", None)
        if tsdf_train is not None:
            tsdf_pred = predictions["tsdf_train"].get("tsdf_pred_train", None)
            tsdf_gt = predictions["tsdf_train"].get("tsdf_gt_train", None)
            tsdf_conf = predictions["tsdf_train"].get("tsdf_conf_train", None)
        else:
            tsdf_pred = None
            tsdf_gt = None
            tsdf_conf = None

        if tsdf_pred is None or tsdf_gt is None:
            # Grid stride subsampling along each spatial dim with random offset per dim
            _, gx, gy, gz, _ = tsdf_xyz.shape
            cfg_step = self.tsdf.get("sample_step", None)
            stride = int(cfg_step) if cfg_step is not None else 1
            stride = max(1, stride)

            # Random offsets in [0, stride)
            offset_x = torch.randint(low=0, high=stride, size=(1,), device=tsdf_xyz.device).item()
            offset_y = torch.randint(low=0, high=stride, size=(1,), device=tsdf_xyz.device).item()
            offset_z = torch.randint(low=0, high=stride, size=(1,), device=tsdf_xyz.device).item()

            tsdf_xyz = tsdf_xyz[:, offset_x::stride, offset_y::stride, offset_z::stride, :]
            tsdf_gt = tsdf[:, offset_x::stride, offset_y::stride, offset_z::stride]

            tsdf_xyz_flatten_gt = tsdf_xyz.reshape(B, -1, 3)
            tsdf_flatten_gt = tsdf_gt.reshape(B, -1)
            # tsdf_t_idx = tsdf ==1.0
            # tsdf_t_idx_flatten = tsdf_t_idx.reshape(B,-1)

            tsdf_pred, tsdf_conf = tsdf_mapper(tsdf_xyz_flatten_gt)  # (B, N)
            tsdf_pred = check_and_fix_inf_nan(tsdf_pred, "tsdf_pred", hard_max=None)
            tsdf_conf = check_and_fix_inf_nan(tsdf_conf, "tsdf_conf", hard_max=None)
        else:
            # tsdf = tsdf_gt
            tsdf_flatten_gt = tsdf_gt.reshape(B, -1)
            # tsdf_t_idx = tsdf ==1.0
            # tsdf_t_idx_flatten = tsdf_t_idx.reshape(B,-1)
            tsdf_pred = check_and_fix_inf_nan(tsdf_pred, "tsdf_pred", hard_max=None)
            tsdf_conf = check_and_fix_inf_nan(tsdf_conf, "tsdf_conf", hard_max=None)
            


        # visualize_tsdf_slices(
        #     tsdf.detach().cpu(),
        #     out_dir="logs/tsdf_slices",
        #     prefix="tsdf",
        #     num_slices=8,
        #     axis="z",
        #     vmax=None,
        # )

        # tsdf_t_idx = tsdf ==1.0
        # tsdf_t_idx_flatten = tsdf_t_idx.reshape(B,-1)

        conf_thershold = self.tsdf.get("conf_thershold", None)
        tsdf_flatten_gt_conf = copy.deepcopy(tsdf_flatten_gt)
        tsdf_flatten_gt_conf[tsdf_flatten_gt == 1.0] = 0.0
        tsdf_flatten_gt_conf[tsdf_flatten_gt != 1.0] = 1.0

        conf_loss = F.huber_loss(tsdf_conf, tsdf_flatten_gt_conf, delta=1.0)

        mask = tsdf_conf >= conf_thershold
        if mask.any():
            tsdf_loss = F.huber_loss(tsdf_pred[mask], tsdf_flatten_gt[mask], delta=1.0)
        else:
            tsdf_loss = torch.tensor(0.0, device=tsdf_pred.device, requires_grad=True)
            if mask.numel() > 0: # Avoid logging if tensor is completely empty for some reason, though unlikely here
                logging.warning("No samples satisfied conf_threshold in compute_sdf_loss, setting tsdf_loss to 0.")
        
        # loss_tsdf = cong_loss + tsdf_loss

        # loss_tsdf = check_and_fix_inf_nan(loss_tsdf, "loss_tsdf", hard_max=None)

        return {"loss_tsdf_conf": conf_loss, "loss_tsdf": tsdf_loss, "pred_tsdf": tsdf_pred}

    # def sdf_cross_attention(self,tsdf_token,tsdf_xyz):
        

    def compute_gs_loss(self,predictions, batch, depth_scale=None, train=True, **kwargs):
        # if "rend_normal" in render_pkg:
        #     render_mode = "2DGS"
        # else:
        #     render_mode = "3DGS"
        render_mode = self.gs['gs_mode']
        assert render_mode in ['2DGS','3DGS','GGGS'], f"render_mode must be one of ['2DGS','3DGS','GGGS'], but got {render_mode}"
        lpips_loss = 0.0
        gs_l1_demo = None
        gs_mse_demo = None
        gs_psnr_demo = None

        if render_mode == '2DGS':
            GS_rendered_color = []
            GS_rendered_depth = []
            GS_rendered_renderednormal = []
            GS_rendered_surfacenormal = []
            GS_rendered_dist = []
            for ii in range(batch['images'].shape[0]):
                GS_rendered_color_batch = []
                GS_rendered_depth_batch = []
                GS_rendered_renderednormal_batch = []
                GS_rendered_surfacenormal_batch = []
                GS_rendered_dist_batch = []
                for i in range(batch['images'].shape[1]):
                    # render_pkg = render(cam_list_all[ii][i], gs_world, gs_pipe, gs_background,batch_idx = ii,sh_degree= self.gs_adapter.sh_degree)
                    render_pkg = predictions['GS_render_pkgs'][ii][i]
                    GS_rendered_color_batch.append(render_pkg['render'].unsqueeze(0))
                    GS_rendered_depth_batch.append(render_pkg['depth'])
                    GS_rendered_renderednormal_batch.append(render_pkg['rend_normal'].unsqueeze(0))
                    GS_rendered_surfacenormal_batch.append(render_pkg['surf_normal'].unsqueeze(0))
                    GS_rendered_dist_batch.append(render_pkg['rend_dist'].unsqueeze(0))
                GS_rendered_color.append(torch.cat(GS_rendered_color_batch,dim=0).unsqueeze(0))
                GS_rendered_depth.append(torch.cat(GS_rendered_depth_batch,dim=0).unsqueeze(0))
                GS_rendered_renderednormal.append(torch.cat(GS_rendered_renderednormal_batch,dim=0).unsqueeze(0))
                GS_rendered_surfacenormal.append(torch.cat(GS_rendered_surfacenormal_batch,dim=0).unsqueeze(0))
                GS_rendered_dist.append(torch.cat(GS_rendered_dist_batch,dim=0).unsqueeze(0))
            GS_rendered_colors = torch.cat(GS_rendered_color,dim=0)
            GS_rendered_depths = torch.cat(GS_rendered_depth,dim=0)
            GS_rendered_renderednormals = torch.cat(GS_rendered_renderednormal,dim=0)
            GS_rendered_surfacenormals = torch.cat(GS_rendered_surfacenormal,dim=0)
            GS_rendered_dists = torch.cat(GS_rendered_dist,dim=0)

            conf = predictions['depth_conf']
            conf_normed = (conf - 1.0) / 100.0 + 1e-6
            # GS_rendered_colors = predictions['GS_rendered_colors']
            # GS_rendered_depths = predictions['GS_rendered_depths']
            gt_images = batch['images']

            # Weight rendered colors by the normalized confidence map per spatial location
            conf_normed_broadcast = conf_normed.unsqueeze(2)
            GS_rendered_colors_confed = GS_rendered_colors * conf_normed_broadcast
            gt_images_confed = gt_images * conf_normed_broadcast
            GS_rendered_renderednormals_confed = GS_rendered_renderednormals * conf_normed_broadcast
            GS_rendered_surfacenormals_confed = GS_rendered_surfacenormals * conf_normed_broadcast
            GS_rendered_dists_confed = GS_rendered_dists * conf_normed_broadcast



            lpips_loss = self.lpips_model.forward(
                GS_rendered_colors_confed.flatten(start_dim=0, end_dim=1),
                gt_images_confed.flatten(start_dim=0, end_dim=1),
            ).mean()
            l1_loss = F.l1_loss(GS_rendered_colors_confed, gt_images_confed)
            ssim_loss = 1.0 - fused_ssim(GS_rendered_colors_confed.squeeze(0), gt_images_confed.squeeze(0))
            
            rgb_loss = (1 - self.gs["lambda_dssim"]) * l1_loss + self.gs["lambda_dssim"] * ssim_loss
            lpips_loss = self.gs.get("lambda_lpips", 0.0) * lpips_loss
            normal_error = (1 - (GS_rendered_renderednormals_confed * GS_rendered_surfacenormals_confed).sum(dim=2))[None]
            normal_loss = self.gs["lambda_normal"] * (normal_error).mean()

            dist_loss = self.gs["lambda_dist"] * (GS_rendered_dists_confed).mean()
            depth_loss = 0.0
            depth_align_loss = 0.0
        elif render_mode == '3DGS':
            GS_rendered_color = []
            for ii in range(batch['images'].shape[0]):
                GS_rendered_color_batch = []
                for i in range(batch['images'].shape[1]):
                    # render_pkg = render(cam_list_all[ii][i], gs_world, gs_pipe, gs_background,batch_idx = ii,sh_degree= self.gs_adapter.sh_degree)
                    render_pkg = predictions['GS_render_pkgs'][ii][i]
                    GS_rendered_color_batch.append(render_pkg['render'].unsqueeze(0))
                GS_rendered_color.append(torch.cat(GS_rendered_color_batch,dim=0).unsqueeze(0))
            GS_rendered_colors = torch.cat(GS_rendered_color,dim=0)

            conf = predictions['depth_conf']
            conf_normed = (conf - 1.0) / 100.0 + 1e-6
            # GS_rendered_colors = predictions['GS_rendered_colors']
            # GS_rendered_depths = predictions['GS_rendered_depths']
            gt_images = batch['images']

            # Weight rendered colors by the normalized confidence map per spatial location
            conf_normed_broadcast = conf_normed.unsqueeze(2)
            GS_rendered_colors_confed = GS_rendered_colors * conf_normed_broadcast
            gt_images_confed = gt_images * conf_normed_broadcast


            lpips_loss = self.lpips_model.forward(
                GS_rendered_colors_confed.flatten(start_dim=0, end_dim=1),
                gt_images_confed.flatten(start_dim=0, end_dim=1),
            ).mean()
            l1_loss = F.l1_loss(GS_rendered_colors_confed, gt_images_confed)
            ssim_loss = 1.0 - fused_ssim(GS_rendered_colors_confed.squeeze(0), gt_images_confed.squeeze(0))
            
            rgb_loss = (1 - self.gs["lambda_dssim"]) * l1_loss + self.gs["lambda_dssim"] * ssim_loss
            lpips_loss = self.gs.get("lambda_lpips", 0.0) * lpips_loss
            normal_loss = 0.0

            dist_loss = 0.0
            depth_loss = 0.0
            depth_align_loss = 0.0
        elif render_mode == 'GGGS':
            GS_rendered_color = []
            GS_rendered_depth = []
            GS_rendered_normal = []
            GS_rendered_alpha = []
            GS_rendered_depthnormal = []
            GS_rendered_validmask = []

            extrinsics = predictions['extrinsics']
            intrinsics = predictions['intrinsics']
            image_h, image_w = batch['images'].shape[-2:]
            last_row = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
            last_row[..., 0, 3] = 1.0
            extrinsics_h = torch.cat([extrinsics, last_row], dim=-2)
            cam_list_all = build_gs_camera(
                K=intrinsics,
                ext=extrinsics_h,
                height=image_h,
                width=image_w,
                data_device=batch['images'].device,
            )

            for ii in range(batch['images'].shape[0]):
                GS_rendered_color_batch = []
                GS_rendered_depth_batch = []
                GS_rendered_normal_batch = []
                GS_rendered_alpha_batch = []
                GS_rendered_depthnormal_batch = []
                GS_rendered_validmask_batch = []
                for i in range(batch['images'].shape[1]):
                    # render_pkg = render(cam_list_all[ii][i], gs_world, gs_pipe, gs_background,batch_idx = ii,sh_degree= self.gs_adapter.sh_degree)
                    render_pkg = predictions['GS_render_pkgs'][ii][i]
                    GS_rendered_color_batch.append(render_pkg['render'].unsqueeze(0))
                    GS_rendered_depth_batch.append(render_pkg['depth'].unsqueeze(0))
                    GS_rendered_normal_batch.append(render_pkg['normal'].unsqueeze(0))
                    GS_rendered_alpha_batch.append(render_pkg['alpha'].unsqueeze(0))

                    depth_normal = depth_to_normal(cam_list_all[ii][i], render_pkg['depth'])
                    depth_normal = depth_normal.permute(2, 0, 1)
                    valid_mask = torch.isfinite(render_pkg['depth']) & (render_pkg['alpha'] > 0)
                    valid_mask = valid_mask & (depth_normal.abs().sum(dim=0, keepdim=True) > 0)

                    GS_rendered_depthnormal_batch.append(depth_normal.unsqueeze(0))
                    GS_rendered_validmask_batch.append(valid_mask.unsqueeze(0))

                GS_rendered_color.append(torch.cat(GS_rendered_color_batch,dim=0).unsqueeze(0))
                GS_rendered_depth.append(torch.cat(GS_rendered_depth_batch,dim=0).unsqueeze(0))
                GS_rendered_normal.append(torch.cat(GS_rendered_normal_batch,dim=0).unsqueeze(0))
                GS_rendered_alpha.append(torch.cat(GS_rendered_alpha_batch,dim=0).unsqueeze(0))
                GS_rendered_depthnormal.append(torch.cat(GS_rendered_depthnormal_batch,dim=0).unsqueeze(0))
                GS_rendered_validmask.append(torch.cat(GS_rendered_validmask_batch,dim=0).unsqueeze(0))
            GS_rendered_colors = torch.cat(GS_rendered_color,dim=0)
            GS_rendered_depths = torch.cat(GS_rendered_depth,dim=0)
            GS_rendered_normals = torch.cat(GS_rendered_normal,dim=0)
            GS_rendered_alphas = torch.cat(GS_rendered_alpha,dim=0)
            GS_rendered_depthnormals = torch.cat(GS_rendered_depthnormal,dim=0)
            GS_rendered_validmasks = torch.cat(GS_rendered_validmask,dim=0)
            GS_rendered_depths = GS_rendered_depths.squeeze(2)
            GS_rendered_alphas = GS_rendered_alphas.squeeze(2)
            if depth_scale is not None:
                GS_rendered_depths = GS_rendered_depths * depth_scale[..., None, None]
            pred_depth = predictions['depth']
            if pred_depth.ndim == 5:
                pred_depth = pred_depth.squeeze(-1)
            if depth_scale is not None:
                pred_depth = pred_depth * depth_scale[..., None, None]

            conf = predictions['depth_conf']
            conf_normed = (conf - 1.0) / 100.0 + 1e-6
            # GS_rendered_colors = predictions['GS_rendered_colors']
            # GS_rendered_depths = predictions['GS_rendered_depths']
            gt_images = batch['images']
            gt_depths = batch['depths']

            # Weight rendered colors by the normalized confidence map per spatial location
            conf_normed_broadcast = conf_normed.unsqueeze(2)
            GS_rendered_colors_confed = GS_rendered_colors * conf_normed_broadcast
            gt_images_confed = gt_images * conf_normed_broadcast

            lpips_loss = self.lpips_model.forward(GS_rendered_colors.flatten(start_dim=0, end_dim=1),gt_images.flatten(start_dim=0, end_dim=1)).mean()
            l1_loss = F.l1_loss(GS_rendered_colors, gt_images)
            
            ssim_loss = 1.0 - fused_ssim(GS_rendered_colors.flatten(0,1), gt_images.flatten(0,1))
            
            rgb_loss = (1 - self.gs["lambda_dssim"]) * l1_loss + self.gs["lambda_dssim"] * ssim_loss
            lpips_loss = self.gs.get("lambda_lpips", 0.0) * lpips_loss
            normal_cos = (GS_rendered_normals * GS_rendered_depthnormals).sum(dim=2).clamp(-1.0, 1.0)
            normal_error = 1.0 - normal_cos
            valid_mask = GS_rendered_validmasks.squeeze(2)
            if valid_mask.any():
                normal_loss = self.gs["lambda_normal"] * normal_error.masked_select(valid_mask).mean()
            else:
                normal_loss = torch.tensor(0.0, device=GS_rendered_colors.device, requires_grad=True)
                # 这里的normal loss是对齐的loss而不是和gt的差异

            gt_depth_valid = batch['point_masks'] & torch.isfinite(gt_depths)
            render_depth_valid = torch.isfinite(GS_rendered_depths) & (GS_rendered_alphas > 0)
            depth_valid_mask = gt_depth_valid & render_depth_valid
            depth_weight = conf_normed
            if depth_valid_mask.any():
                depth_error = (GS_rendered_depths - gt_depths).abs()
                depth_loss = self.gs.get("lambda_depth", 1.0) * ((depth_error * depth_weight)[depth_valid_mask]).mean()
            else:
                depth_loss = torch.tensor(0.0, device=GS_rendered_colors.device, requires_grad=True)

            depth_align_valid_mask = torch.isfinite(pred_depth) & torch.isfinite(GS_rendered_depths) & (GS_rendered_alphas > 0)
            if depth_align_valid_mask.any():
                depth_align_error = (pred_depth - GS_rendered_depths).abs()
                depth_align_loss = self.gs.get("lambda_depth_align", 1.0) * ((depth_align_error * depth_weight)[depth_align_valid_mask]).mean()
            else:
                depth_align_loss = torch.tensor(0.0, device=GS_rendered_colors.device, requires_grad=True)
            
            

            dist_loss = 0.0
        else:
            lpips_loss = 0.0
            depth_loss = 0.0
            depth_align_loss = 0.0

        if "GS_rendered_colors" in locals() and "gt_images" in locals():
            gs_rendered_demo = GS_rendered_colors.float().clamp(0.0, 1.0)
            gt_images_demo = gt_images.float().clamp(0.0, 1.0)
            demo_abs_error = (gs_rendered_demo - gt_images_demo).abs().flatten(start_dim=2)
            demo_squared_error = (gs_rendered_demo - gt_images_demo).square().flatten(start_dim=2)
            per_image_l1 = demo_abs_error.mean(dim=-1)
            per_image_mse = demo_squared_error.mean(dim=-1)
            gs_l1_demo = per_image_l1.mean()
            gs_mse_demo = per_image_mse.mean()
            if not train:
                gs_psnr_demo = (20.0 * torch.log10(1.0 / torch.sqrt(per_image_mse.clamp_min(1e-10)))).mean()

        return {
            "loss_gs_rgb": rgb_loss,
            "loss_gs_lpips": lpips_loss,
            "loss_gs_normal": normal_loss,
            "loss_gs_dist": dist_loss,
            "loss_gs_depth": depth_loss,
            "depth_align_loss": depth_align_loss,
            "loss_gs_l1_demo": gs_l1_demo,
            "loss_gs_mse_demo": gs_mse_demo,
            "psnr_gs_demo": gs_psnr_demo,
        }

def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    depth_scale = None,
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    if depth_scale is not None:
        pred_pose_encodings = [
            torch.cat((pose_enc[..., :3] * depth_scale[..., None], pose_enc[..., 3:]), dim=-1)
            for pose_enc in pred_pose_encodings
        ]
    if "camera_valid_mask" in batch_data:
        valid_frame_mask = batch_data["camera_valid_mask"].bool()
        if valid_frame_mask.ndim > 1:
            valid_frame_mask = valid_frame_mask.view(valid_frame_mask.shape[0], -1).any(dim=1)
    else:
        # Binary mask for valid points per frame (B, N, H, W)
        point_masks = batch_data['point_masks']
        # Only consider frames with enough valid points (>100)
        valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0: 
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions.get('world_points_conf', predictions.get('depth_conf', torch.ones_like(pred_points[..., 0])))
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, align_scale=False, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict, None

    # Scale-invariant alignment: align pred_depth to gt_depth scale before loss,
    # consistent with amb3r/training.py scale_invariant_alignment usage.
    depth_scale = None
    if align_scale:
        bs, t, h, w = gt_depth_mask.shape  # mask is always (B, S, H, W)
        # pred_depth may be (B, S, H, W) or (B, S, H, W, 1); ensure 5D with channel dim
        pred_depth_5d = pred_depth if pred_depth.ndim == 5 else pred_depth[..., None]
        # Use (B*t, h, w, 3) so each frame is aligned independently;
        # avoids distorted aspect ratio from the old (B, t*h, w, 3) layout
        pred_depth_5d, depth_scale = scale_invariant_alignment(
            pred_depth_5d.repeat(1, 1, 1, 1, 3),
            gt_depth.repeat(1, 1, 1, 1, 3),
            gt_depth_mask,
            trunc=None, detach=False
        )
        pred_depth = pred_depth_5d.view(bs, t, h, w, 3)[..., :1]

        

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict,depth_scale



def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask] + 1e-6)
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf + 1e-6)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x + 1e-6)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y + 1e-6)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast('cuda', enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''
