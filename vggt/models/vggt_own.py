# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from types import SimpleNamespace

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.gsdpt import GSDPT
from vggt.heads.gs_adapter import GaussianAdapter

from vggt.heads.vsdf_head import VSDFHead

from vggt.utils.specs import Gaussians

from vggt.utils.geometry import map_pdf_to_opacity
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.gsply_helpers import save_gaussian_ply
from vggt.utils.sh_helpers import RGB2SH

from vggt.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode,render_3dgs

from vggt.models.GS.utils.build_camera import build_gs_camera  
from vggt.models.GS.gaussian_renderer import GaussianModel,render, network_gui

from tqdm import tqdm
import open3d as o3d
import numpy as np


# 这里就是正经的VGGT模型
class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_gs=True,gs_options=None, enable_vsdf=False, vsdf_options=None):
        super().__init__()
        
        # self.chunk_size = chunk_size

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.gsdpt_head = GSDPT(dim_in=2 * embed_dim, activation="linear", conf_activation="sigmoid",gs_options = gs_options) if enable_gs else None
        self.gs_adapter = GaussianAdapter(pred_color=False, pred_offset_depth =False, pred_offset_xy=False,gaussian_scale_min=1e-05,gaussian_scale_max=30.0,gs_options = gs_options) if enable_gs else None
    
        self.vsdf_head = VSDFHead(dim_in=2 * embed_dim, vsdf_options=vsdf_options) if enable_vsdf else None
    def to(self, *args, **kwargs):
        # TODO: this won't work if the module is inside another module
        self.aggregator = self.aggregator.to(*args, **kwargs) if self.aggregator is not None else None
        self.camera_head = self.camera_head.to(*args, **kwargs) if self.camera_head is not None else None
        self.point_head = self.point_head.to(*args, **kwargs) if self.point_head is not None else None
        self.depth_head = self.depth_head.to(*args, **kwargs) if self.depth_head is not None else None
        self.track_head = self.track_head.to(*args, **kwargs) if self.track_head is not None else None
        self.gsdpt_head = self.gsdpt_head.to(*args, **kwargs) if self.gsdpt_head is not None else None
        self.gs_adapter = self.gs_adapter.to(*args, **kwargs) if self.gs_adapter is not None else None
        self.vsdf_head = self.vsdf_head.to(*args, **kwargs) if self.vsdf_head is not None else None
        return self
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,verbose = False,forward_dict = None,gt_data = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape
        # print("Images shape:",images.shape)
        # print("Batch size:",B)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images,verbose = verbose,chunk_size=128)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                if verbose:
                    print("Running camera head")
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                # predictions["aggregated_tokens_list"] = aggregated_tokens_list
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], image_size_hw=(H, W)
                )
                predictions["extrinsics"] = extrinsics
                predictions["intrinsics"] = intrinsics
            if self.depth_head is not None:
                if verbose:
                    print("Running depth head")
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
            if self.gsdpt_head is not None:
                if verbose:
                    print("Running GS head")
                
                sh = RGB2SH(images).permute(0,1,3,4,2) 
                raw_gaussians, densities = self.gsdpt_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                # raw_gaussians = gs_outs['raw_gs']
                # densities = gs_outs['raw_gs_conf']
                predictions["raw_gaussians"] = raw_gaussians
                adapter_chunk_size = 1
                # gs_world = []
                if densities.shape[1]>30:
                    print("Eval, Using incremental cpu GS adapter to save memory. Chunk size:",adapter_chunk_size)
                    harmonics = torch.tensor([])
                    means = torch.tensor([])
                    rotations = torch.tensor([])
                    scales = torch.tensor([])
                    opacities = torch.tensor([])
                    cloud_before = None
                    for idx in tqdm(range(0,densities.shape[1],adapter_chunk_size)):
                        depth_conf = predictions["depth_conf"][:,idx:min(idx+adapter_chunk_size,densities.shape[1])]
                        gs_world_batch = self.gs_adapter(
                            depths=depth.squeeze(-1)[:,idx:min(idx+adapter_chunk_size,densities.shape[1])],
                            opacities=map_pdf_to_opacity(densities)[:,idx:min(idx+adapter_chunk_size,densities.shape[1])],
                            raw_gaussians=raw_gaussians[:,idx:min(idx+adapter_chunk_size,densities.shape[1])],
                            image_shape=(H, W),
                            extrinsics=extrinsics[:,idx:min(idx+adapter_chunk_size,densities.shape[1])],
                            intrinsics=intrinsics[:,idx:min(idx+adapter_chunk_size,densities.shape[1])],
                            sh_RGB = sh[:,idx:min(idx+adapter_chunk_size,densities.shape[1])]
                        ) 
                        
                        if cloud_before is None:
                            depth_conf_flatten = depth_conf.flatten().cpu().numpy() > forward_dict['conf_thres_value']
                            mask = np.ones(gs_world_batch.means.shape[1], dtype=bool)
                            whole_mask = np.logical_and(mask, depth_conf_flatten)
                            # 初始化场景点云，只保留通过置信度过滤的点
                            whole_mask_tensor_init = torch.from_numpy(whole_mask).bool()
                            cloud_before = gs_world_batch.means.cpu()[:, whole_mask_tensor_init, :]
                        else:
                            # 用整个已累积场景构建KDTree，对当前帧每个点查询最近邻
                            pcd_scene = o3d.geometry.PointCloud()
                            pcd_scene.points = o3d.utility.Vector3dVector(np.array(cloud_before.numpy()[0]))
                            pcd_scene_tree = o3d.geometry.KDTreeFlann(pcd_scene)
                            current_points = gs_world_batch.means.cpu().numpy()[0]
                            dist_list = []
                            for point in current_points:
                                k, index, distance = pcd_scene_tree.search_knn_vector_3d(point, 1)
                                dist = np.sqrt(distance[0])
                                dist_list.append(dist)
                            dist_array = np.array(dist_list)
                            threshold = forward_dict['dist_threshold']  # 设置距离阈值
                            mask = dist_array > threshold
                            depth_conf_flatten = depth_conf.flatten().cpu().numpy() > forward_dict['conf_thres_value']
                            mask_wo_badconf = mask[depth_conf_flatten]

                            whole_mask = np.logical_and(mask, depth_conf_flatten)

                            chongdielv = 1.0 - mask_wo_badconf.sum() / mask_wo_badconf.shape[0]

                            chongdielv_thereshold = forward_dict['overlap_threshold']
                            if chongdielv > chongdielv_thereshold:
                                print(f"{idx} 到 {min(idx+adapter_chunk_size,densities.shape[1])} 的高斯重叠率为 {chongdielv} 跳过")
                                continue
                            else:
                                print(f"接受 {idx} 到 {min(idx+adapter_chunk_size,densities.shape[1])} 的高斯 重叠率为:", chongdielv)
                                # 重叠率低，仅按置信度过滤，把所有置信度合格的点加进去
                                whole_mask = depth_conf_flatten
                                whole_mask_tensor_temp = torch.from_numpy(whole_mask).bool()
                                cloud_before = torch.cat([cloud_before, gs_world_batch.means.cpu()[:, whole_mask_tensor_temp, :]], dim=1)
                            
                        mask_tensor = torch.from_numpy(mask).bool()
                        whole_mask_tensor = torch.from_numpy(whole_mask).bool() 
                        harmonics = torch.cat([harmonics, gs_world_batch.harmonics.cpu()[:, whole_mask_tensor, :, :]], dim=1)
                        means = torch.cat([means, gs_world_batch.means.cpu()[:, whole_mask_tensor, :]], dim=1)
                        rotations = torch.cat([rotations, gs_world_batch.rotations.cpu()[:, whole_mask_tensor, :]], dim=1)
                        scales = torch.cat([scales, gs_world_batch.scales.cpu()[:, whole_mask_tensor, :]], dim=1)
                        opacities = torch.cat([opacities, gs_world_batch.opacities.cpu()[:, whole_mask_tensor]], dim=1)

                        del gs_world_batch
                        torch.cuda.empty_cache()
                        # break
                        
                        # gs_world.append(gs_world_batch)
                    gs_world = Gaussians(
                        means=means,
                        scales=scales,
                        rotations=rotations,
                        harmonics=harmonics,
                        opacities=opacities
                    )
                else:
                    gs_world = self.gs_adapter(
                        depths=depth.squeeze(-1),
                        opacities=map_pdf_to_opacity(densities),
                        raw_gaussians=raw_gaussians,
                        image_shape=(H, W),
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        sh_RGB = sh
                    )

                predictions["gs_world"] = gs_world
                
                if False:
                    depth_conf_mask = (predictions["depth_conf"] > 5.0).squeeze(0)
                    gs_views_interval = max(predictions["depth"].shape[0] // 12, 1)
                    save_gaussian_ply(
                        gaussians=gs_world,
                        save_path="test.ply",
                        ctx_depth=depth[0],
                        shift_and_scale=False,
                        save_sh_dc_only=True,
                        gs_views_interval=gs_views_interval,
                        inv_opacity=True,
                        prune_by_depth_percent=0.9,
                        prune_border_gs=True,
                        match_3dgs_mcmc_dev=False,
                        conf_mask=depth_conf_mask
                    )


                

                last_row = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
                last_row[..., 0, 3] = 1.0
                extrinsics_h = torch.cat([extrinsics, last_row], dim=-2)
                
                intr_normed = intrinsics.clone().detach()
                intr_normed[..., 0, :] /= W
                intr_normed[..., 1, :] /= H


                





                # GS_rendered_color = []
                # GS_rendered_depth = []
                # for i in range(images.shape[1]):
                #     color, depth = render_3dgs(
                #             gaussian=gs_world,
                #             extrinsics=extrinsics_h[:,i],
                #             intrinsics=intr_normed[:,i],
                #             image_shape=(H, W),
                #             trj_mode="extend",
                #             use_sh=True,
                #             color_mode='RGB+ED',
                #         )# 基于gsplat
                    
                #     GS_rendered_color.append(color)
                #     GS_rendered_depth.append(depth)
                # GS_rendered_colors = torch.stack(GS_rendered_color,dim=1)
                # GS_rendered_depths = torch.stack(GS_rendered_depth,dim=1)
                # predictions["GS_rendered_colors"] = GS_rendered_colors
                # predictions["GS_rendered_depths"] = GS_rendered_depths



                cam_list_all = build_gs_camera(K = intrinsics,ext = extrinsics_h, height=H,width=W,data_device=images.device)
                gs_background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
                gs_pipe = SimpleNamespace(
                    convert_SHs_python=False,
                    compute_cov3D_python=False,
                    depth_ratio=0.0,
                    kernel_size=0.0,
                    require_depth=True,
                    debug=False,
                )
                # GS_rendered_color = []
                # GS_rendered_depth = []
                render_pkgs = []
                for ii in range(images.shape[0]):
                    # GS_rendered_color_batch = []
                    # GS_rendered_depth_batch = []
                    render_pkgs_batch = []
                    for i in range(images.shape[1]):
                        render_pkg = render(cam_list_all[ii][i], gs_world, gs_pipe, gs_background,batch_idx = ii,sh_degree= self.gs_adapter.sh_degree)
                        render_pkgs_batch.append(render_pkg)
                        # GS_rendered_color_batch.append(render_pkg['render'].unsqueeze(0))
                        # GS_rendered_depth_batch.append(render_pkg['surf_depth'])
                    # GS_rendered_color.append(torch.cat(GS_rendered_color_batch,dim=0).unsqueeze(0))
                    # GS_rendered_depth.append(torch.cat(GS_rendered_depth_batch,dim=0).unsqueeze(0))
                    render_pkgs.append(render_pkgs_batch)
                # GS_rendered_colors = torch.cat(GS_rendered_color,dim=0)
                # GS_rendered_depths = torch.cat(GS_rendered_depth,dim=0)
                predictions["GS_render_pkgs"] = render_pkgs
                # predictions["GS_rendered_depths"] = GS_rendered_depths

            if self.vsdf_head is not None:
                if verbose:
                    print("Running VSDF head")
                tsdf_token, tsdf_mapper = self.vsdf_head(
                    predictions, aggregated_tokens_list,images=images, patch_start_idx=patch_start_idx,gt_data = gt_data
                )
                predictions["tsdf_token"] = tsdf_token  # vsdf outputs of the last iteration
                predictions["tsdf_mapper"] = tsdf_mapper
            if self.track_head is not None and query_points is not None:
                if verbose:
                    print("Running Tracking head")
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf

            if not self.training:
                predictions["images"] = images  # store the images for visualization during inference

        return predictions

