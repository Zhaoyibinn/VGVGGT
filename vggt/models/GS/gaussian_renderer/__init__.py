#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2D, GaussianRasterizer as GaussianRasterizer2D
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings3D, GaussianRasterizer as GaussianRasterizer3D
from gggs_diff_gaussian_rasterization import GaussianRasterizationSettings as gggs_GaussianRasterizationSettings, GaussianRasterizer as gggs_GaussianRasterizer
from scene.gaussian_model import GaussianModel
from GS.utils.sh_utils import eval_sh
from GS.utils.point_utils import depth_to_normal




def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,batch_idx=0,sh_degree = 0, gs_mode = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    xyz = pc.means[batch_idx]
    opacity = pc.opacities[batch_idx].unsqueeze(-1)
    # scales = pc.scales[batch_idx]
    # rotations = pc.rotations[batch_idx]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if gs_mode is not None:
        requested_render_mode = gs_mode
    else:
        if pc.scales[batch_idx].shape[1] == 2:
            requested_render_mode = "2DGS"
        else:
            requested_render_mode = "3DGS"

    
    if requested_render_mode == "GGGS":
        GaussianRasterizationSettings = gggs_GaussianRasterizationSettings
        GaussianRasterizer = gggs_GaussianRasterizer
        render_mode = requested_render_mode
    elif requested_render_mode == "2DGS":
        GaussianRasterizationSettings = GaussianRasterizationSettings2D
        GaussianRasterizer = GaussianRasterizer2D
        render_mode = requested_render_mode
    else:
        GaussianRasterizationSettings = GaussianRasterizationSettings3D
        GaussianRasterizer = GaussianRasterizer3D
        render_mode = "3DGS"

    if render_mode == "GGGS":
        kernel_size = float(getattr(pipe, "kernel_size", 0.0))
        sg_degree = 0
        require_depth = bool(getattr(pipe, "require_depth", True))
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            sg_degree=sg_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            require_depth=require_depth,
            debug=bool(getattr(pipe, "debug", False)),
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing = False
            # pipe.debug
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.scales[batch_idx]
        rotations = pc.rotations[batch_idx]
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # shs = pc.get_features
            shs = pc.harmonics[batch_idx].permute(0,2,1)
    else:
        colors_precomp = override_color
    if "cuda" not in str(means3D.device):
        # print("由于是稠密的高斯 保存位置为cpu 这里就不渲染了")
        # rendered_image, radii, allmap = None, None, None
        rets =  {"render": None,
                "viewspace_points": None,
                "visibility_filter" : None,
                "radii": None,
        }
    else:   
        if render_mode == "2DGS":
            rendered_image, radii, allmap = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp
            )
        
            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            rets =  {"render": rendered_image,
                    "viewspace_points": means2D,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
            }


            # additional regularizations
            render_alpha = allmap[1:2]

            # get normal map
            # transform normal from view space to world space
            render_normal = allmap[2:5]
            render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
            
            # get median depth map
            render_depth_median = allmap[5:6]
            render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

            # get expected depth map
            render_depth_expected = allmap[0:1]
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
            
            # get depth distortion map
            render_dist = allmap[6:7]

            # psedo surface attributes
            # surf depth is either median or expected by setting depth_ratio to 1 or 0
            # for bounded scene, use median depth, i.e., depth_ratio = 1; 
            # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
            surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
            
            # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2,0,1)
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal * (render_alpha).detach()


            rets.update({
                    'depth': surf_depth,
                    'rend_alpha': render_alpha,
                    'rend_normal': render_normal,
                    'rend_dist': render_dist,
                    'surf_depth': surf_depth,
                    'surf_normal': surf_normal,
            })
        elif render_mode == "3DGS":
            rendered_image, radii, inv_depth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp
            )

            depth = 1.0 / inv_depth.clamp(min=1e-6)
            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            rets =  {"render": rendered_image,
                    "depth": depth,
                    "viewspace_points": means2D,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
            }
        elif render_mode == "GGGS":
            def _get_optional_pc_attr_gggs(pc, attr_name, batch_idx=0, default=None):
                if not hasattr(pc, attr_name):
                    return default
                value = getattr(pc, attr_name)
                value = value() if callable(value) else value
                if torch.is_tensor(value) and value.ndim >= 3:
                    return value[batch_idx]
                return value
            sg_axis = _get_optional_pc_attr_gggs(pc, "sg_axis", batch_idx=batch_idx)
            sg_sharpness = _get_optional_pc_attr_gggs(pc, "sg_sharpness", batch_idx=batch_idx)
            sg_color = _get_optional_pc_attr_gggs(pc, "sg_color", batch_idx=batch_idx)
            empty_tensor = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
            if sg_axis is None:
                sg_axis = empty_tensor
            if sg_sharpness is None:
                sg_sharpness = empty_tensor
            if sg_color is None:
                sg_color = empty_tensor

            rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                sg_axis = sg_axis,
                sg_sharpness = sg_sharpness,
                sg_color = sg_color,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3Ds_precomp = cov3D_precomp,
            )

            rets = {
                "render": rendered_image,
                "depth": rendered_median_depth,
                "alpha": rendered_alpha,
                "median_depth": rendered_median_depth,
                "viewspace_points": means2D,
                "visibility_filter": radii > 0,
                "radii": radii,
                "normal": rendered_normal,
            }

    return rets
