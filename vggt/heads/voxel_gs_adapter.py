import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from einops import einsum
from vggt.models.depth_anything_3.specs import Gaussians

from vggt.dependency.pointMLP import pointMLPEncoderBase6, pointMLPEncoderBasezyb1

class VoxelGSAdapter(nn.Module):
    def __init__(
        self,
        backend_out_dim: int,
        sh_degree: int = 2,
        pred_color: bool = False,
        add_point_color_residual: bool = True,
        gaussian_scale_min: float = 1e-05,
        gaussian_scale_max: float = 30.0,
        gaussian_offset_max: Optional[float] = None,
        gaussian_splits: int = 1,
        gs_options = None,
        backbone_dim = 1024
    ):
        super().__init__()
        self.gs_options = gs_options
        self.out_dim = backend_out_dim
        self.sh_degree = self.gs_options["sh_degree"] if self.gs_options is not None else sh_degree
        self.pred_color = pred_color
        self.add_point_color_residual = add_point_color_residual
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.gaussian_offset_max = (
            (gaussian_scale_max - gaussian_scale_min)
            if gaussian_offset_max is None
            else gaussian_offset_max
        )
        self.gaussian_splits = int(gaussian_splits)
        if self.gaussian_splits < 1:
            raise ValueError(f"gaussian_splits must be >= 1, got {self.gaussian_splits}")
        self.backbone_dim = backbone_dim
        self.scale_dim = 2 if (self.gs_options is not None and self.gs_options.get('gs_mode') == "2DGS") else 3
        self.d_sh = (self.sh_degree + 1) ** 2

        self.pointMLP = pointMLPEncoderBasezyb1(feature_channel=backend_out_dim, batch_norm=False, output_dim = backbone_dim)
        if not self.pred_color:
            self.register_buffer(
                "sh_mask",
                torch.ones((self.d_sh,), dtype=torch.float32),
                persistent=False,
            )
            for degree in range(1, sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree


        backend_gs_out_dim = 3 + self.scale_dim + 4 + 3 * self.d_sh + 1
        
        self.shared = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim * 2),
            nn.GELU(),
            nn.Linear(backbone_dim * 2,backbone_dim),
            nn.GELU(),
        )
        self.xyz_head = nn.Linear(backbone_dim, 3 * self.gaussian_splits)
        self.sh_head = nn.Linear(backbone_dim, 3 * self.d_sh * self.gaussian_splits)
        self.scale_head = nn.Linear(backbone_dim, self.scale_dim * self.gaussian_splits)
        self.rotation_head = nn.Linear(backbone_dim, 4 * self.gaussian_splits)
        self.opacity_head = nn.Linear(backbone_dim, self.gaussian_splits)
        
        self.xyz_gate = nn.Parameter(torch.full((1,), 1e-4))
        # self.sh_gate = nn.Parameter(torch.full((1,), 1e-4))
        
        # Only zero-initialize xyz_residual forecasting
        nn.init.zeros_(self.xyz_head.weight)
        nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.sh_head.weight)
        nn.init.zeros_(self.sh_head.bias)

        self.layernorm = nn.LayerNorm(backbone_dim)
        
    # @staticmethod
    # def map_to_args_to_float(args, kwargs):
    #     args = tuple(
    #         torch.float32 if isinstance(arg, torch.dtype) else arg
    #         for arg in args
    #     )
    #     kwargs = dict(kwargs)
    #     for key in kwargs:
    #         if key == "dtype":
    #             kwargs[key] = torch.float32
    #     return args, kwargs

    # def to(self, *args, **kwargs):
    #     self.shared = self.shared.to(*args, **kwargs)

    #     # Move non-persistent sh_mask buffer (must be on same device as other tensors for DDP)
    #     if hasattr(self, 'sh_mask') and self.sh_mask is not None:
    #         self.sh_mask = self.sh_mask.to(*args, **kwargs)

    #     args, kwargs = self.map_to_args_to_float(args, kwargs)
    #     self.xyz_head = self.xyz_head.to(*args, **kwargs)
    #     self.sh_head = self.sh_head.to(*args, **kwargs)
    #     self.scale_head = self.scale_head.to(*args, **kwargs)
    #     self.rotation_head = self.rotation_head.to(*args, **kwargs)
    #     self.opacity_head = self.opacity_head.to(*args, **kwargs)

    #     self.xyz_gate = nn.Parameter(self.xyz_gate.to(*args, **kwargs))
    #     self.sh_gate = nn.Parameter(self.sh_gate.to(*args, **kwargs))

    #     return self

    def decode(self, x):
        h = self.shared(x)
        batch_size = h.shape[0]

        xyz_res = self.xyz_gate * self.xyz_head(h)
        xyz_res = xyz_res.view(batch_size, self.gaussian_splits, 3)

        # sh_res = self.sh_gate * self.sh_head(h)
        sh_res = self.sh_head(h)
        sh_res = sh_res.view(batch_size, self.gaussian_splits, 3, self.d_sh)

        scale = self.scale_head(h)
        scale = scale.view(batch_size, self.gaussian_splits, self.scale_dim)

        rotation = self.rotation_head(h)
        rotation = rotation.view(batch_size, self.gaussian_splits, 4)

        opacity = self.opacity_head(h)
        opacity = opacity.view(batch_size, self.gaussian_splits)

        return xyz_res, scale, rotation, sh_res, opacity

    def _pack_voxel_gaussians(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        harmonics: torch.Tensor,
        opacities: torch.Tensor,
        voxel_conf: Optional[torch.Tensor],
        batch_ids: torch.Tensor,
        batch_size: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        max_count = 1
        counts = []
        for batch_idx in range(batch_size):
            count = int((batch_ids == batch_idx).sum().item())
            counts.append(count)
            max_count = max(max_count, count)

        means_list = []
        scales_list = []
        rotations_list = []
        harmonics_list = []
        opacities_list = []
        voxel_conf_list = [] if voxel_conf is not None else None

        for batch_idx in range(batch_size):
            curr_mask = batch_ids == batch_idx
            curr_count = counts[batch_idx]
            
            m = means[curr_mask]
            s = scales[curr_mask]
            r = rotations[curr_mask]
            h = harmonics[curr_mask]
            o = opacities[curr_mask]
            vc = voxel_conf[curr_mask] if voxel_conf is not None else None

            pad_len = max_count - curr_count
            if pad_len > 0:
                m = torch.cat([m, m.new_zeros(pad_len, *m.shape[1:])], dim=0)
                s = torch.cat([s, s.new_zeros(pad_len, *s.shape[1:])], dim=0)
                r = torch.cat([r, r.new_zeros(pad_len, *r.shape[1:])], dim=0)
                h = torch.cat([h, h.new_zeros(pad_len, *h.shape[1:])], dim=0)
                o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
                if vc is not None:
                    vc = torch.cat([vc, vc.new_zeros(pad_len, *vc.shape[1:])], dim=0)

            means_list.append(m)
            scales_list.append(s)
            rotations_list.append(r)
            harmonics_list.append(h)
            opacities_list.append(o)
            if voxel_conf_list is not None:
                voxel_conf_list.append(vc)

        means_padded = torch.stack(means_list)
        scales_padded = torch.stack(scales_list)
        rotations_padded = torch.stack(rotations_list)
        harmonics_padded = torch.stack(harmonics_list)
        opacities_padded = torch.stack(opacities_list)
        voxel_conf_padded = torch.stack(voxel_conf_list) if voxel_conf_list is not None else None

        return means_padded, scales_padded, rotations_padded, harmonics_padded, opacities_padded, voxel_conf_padded


    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,
        pixel_size: torch.Tensor,
        multiplier: float = 0.1,
    ) -> torch.Tensor:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        last_backend_voxel_detail: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if output.get("depth", None) is None or last_backend_voxel_detail is None:
            return output

        voxel_detail = last_backend_voxel_detail

        voxel_feat = voxel_detail["voxel_feat"]
        voxel_centers = voxel_detail["voxel_centers"]
        voxel_batch_ids = voxel_detail["voxel_batch_ids"]
        voxel_colors = voxel_detail.get("voxel_colors", None)
        voxel_depth_conf = voxel_detail.get("voxel_depth_conf", None)
        
        batch_size = output["depth"].shape[0]

        # Calculate scale multiplier similar to gs_adapter
        depth_output = output["depth"]
        if depth_output.ndim >= 2:
            H, W = depth_output.shape[-2:]
        else:
            H, W = 512, 512
        
        pixel_size = 1 / torch.tensor((W, H), dtype=depth_output.dtype, device=depth_output.device)
        intrinsics = output.get("intrinsics", None)
        if intrinsics is not None:
            intr_normed = intrinsics.clone().detach().float()
            intr_normed[..., 0, :] /= W
            intr_normed[..., 1, :] /= H
            multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        else:
            multiplier = None

        if isinstance(voxel_feat, list):
            means_list = []
            scales_list = []
            rotations_list = []
            harmonics_list = []
            opacities_list = []
            gaussian_voxel_conf_list = [] if voxel_depth_conf is not None else None

            for batch_idx in range(batch_size):
                feat_b = voxel_feat[batch_idx] if batch_idx < len(voxel_feat) else None
                centers_b = voxel_centers[batch_idx] if batch_idx < len(voxel_centers) else None
                colors_b = voxel_colors[batch_idx] if voxel_colors is not None and batch_idx < len(voxel_colors) else None
                depth_conf = voxel_depth_conf[batch_idx] if voxel_depth_conf is not None and batch_idx < len(voxel_depth_conf) else None
                if feat_b is None or feat_b.numel() == 0:
                    means_list.append(torch.zeros((0, 3), device=output["depth"].device))
                    scales_list.append(torch.zeros((0, self.scale_dim), device=output["depth"].device))
                    rotations_list.append(torch.zeros((0, 4), device=output["depth"].device))
                    harmonics_list.append(torch.zeros((0, 3, self.d_sh), device=output["depth"].device))
                    opacities_list.append(torch.zeros((0,), device=output["depth"].device))
                    if gaussian_voxel_conf_list is not None:
                        gaussian_voxel_conf_list.append(torch.zeros((0,), device=output["depth"].device))
                    continue
                # with torch.cuda.amp.autocast(enabled=False):
                pointmlp_feature = self.pointMLP(centers_b.permute(1,0).unsqueeze(0), feat_b.permute(1,0).unsqueeze(0)).squeeze().permute(1,0)
                layernormed_feature = self.layernorm(pointmlp_feature)
                xyz_residual, scales_raw, rotations_raw, harmonics_b, opacity_raw = self.decode(layernormed_feature)
                
                means_b = centers_b[:, None, :] + self.gaussian_offset_max * (xyz_residual.sigmoid()-0.5)
                scales_b = self.gaussian_scale_min + (
                    self.gaussian_scale_max - self.gaussian_scale_min
                ) * scales_raw.sigmoid()
                
                # Apply scale multiplier logic
                # if multiplier is not None:
                #     depths_b = centers_b[..., 2].clamp(min=1e-3)
                #     mult = multiplier[batch_idx]
                #     if mult.ndim > 0:
                #         mult = mult.mean()
                #     scales_b = scales_b * depths_b[:, None, None] * mult
                rotations_b = rotations_raw / (rotations_raw.norm(dim=-1, keepdim=True) + 1e-8)
                if colors_b is not None and self.add_point_color_residual:
                    sh0_b = (colors_b - 0.5) / 0.28209479177387814
                    harmonics_b = harmonics_b.clone()
                    harmonics_b[..., 0] = harmonics_b[..., 0] + sh0_b[:, None, :]
                    # harmonics_b = torch.zeros_like(harmonics_b)
                    # harmonics_b[..., 0] = sh0_b[:, None, :]
                if getattr(self, "pred_color", True) is False:
                    harmonics_b = harmonics_b * self.sh_mask.view(1, 1, 1, -1).to(harmonics_b)
                
                if depth_conf is not None:
                    opacity_raw = opacity_raw + (depth_conf.view(-1, 1) - 5).clamp_(-5, 5)
                opacities_b = opacity_raw.sigmoid()

                means_b = means_b.reshape(-1, 3)
                scales_b = scales_b.reshape(-1, self.scale_dim)
                rotations_b = rotations_b.reshape(-1, 4)
                harmonics_b = harmonics_b.reshape(-1, 3, self.d_sh)
                opacities_b = opacities_b.reshape(-1)
                if depth_conf is not None:
                    gaussian_voxel_conf_b = depth_conf.repeat_interleave(self.gaussian_splits, dim=0).reshape(-1)

                means_list.append(means_b)
                scales_list.append(scales_b)
                rotations_list.append(rotations_b)
                harmonics_list.append(harmonics_b)
                opacities_list.append(opacities_b)
                if gaussian_voxel_conf_list is not None:
                    gaussian_voxel_conf_list.append(gaussian_voxel_conf_b)

            max_count = max([m.shape[0] for m in means_list] + [1])
            means_padded_list = []
            scales_padded_list = []
            rotations_padded_list = []
            harmonics_padded_list = []
            opacities_padded_list = []
            gaussian_voxel_conf_padded_list = []

            for batch_idx in range(batch_size):
                m_b = means_list[batch_idx]
                s_b = scales_list[batch_idx]
                r_b = rotations_list[batch_idx]
                h_b = harmonics_list[batch_idx]
                o_b = opacities_list[batch_idx]
                vc_b = gaussian_voxel_conf_list[batch_idx]
                
                curr_count = m_b.shape[0]
                pad_len = max_count - curr_count

                if pad_len > 0:
                    m_b = torch.cat([m_b, m_b.new_zeros(pad_len, *m_b.shape[1:])], dim=0)
                    s_b = torch.cat([s_b, s_b.new_zeros(pad_len, *s_b.shape[1:])], dim=0)
                    r_b = torch.cat([r_b, r_b.new_zeros(pad_len, *r_b.shape[1:])], dim=0)
                    h_b = torch.cat([h_b, h_b.new_zeros(pad_len, *h_b.shape[1:])], dim=0)
                    o_b = torch.cat([o_b, o_b.new_zeros(pad_len, *o_b.shape[1:])], dim=0)
                    vc_b = torch.cat([vc_b, vc_b.new_zeros(pad_len, *vc_b.shape[1:])], dim=0)

                means_padded_list.append(m_b)
                scales_padded_list.append(s_b)
                rotations_padded_list.append(r_b)
                harmonics_padded_list.append(h_b)
                opacities_padded_list.append(o_b)
                gaussian_voxel_conf_padded_list.append(vc_b)

            means = torch.stack(means_padded_list)
            scales = torch.stack(scales_padded_list)
            rotations = torch.stack(rotations_padded_list)
            harmonics = torch.stack(harmonics_padded_list)
            opacities = torch.stack(opacities_padded_list)
            gaussian_voxel_conf = (
                torch.stack(gaussian_voxel_conf_padded_list)
                if gaussian_voxel_conf_list is not None
                else None
            )

        else:
            voxel_batch_ids = voxel_batch_ids.long()
            xyz_residual, scales_raw, rotations_raw, harmonics, opacity_raw = self.decode(voxel_feat)

            means = voxel_centers[:, None, :] + xyz_residual
            scales = self.gaussian_scale_min + (
                self.gaussian_scale_max - self.gaussian_scale_min
            ) * scales_raw.sigmoid()
            
            if multiplier is not None:
                depths_flat = voxel_centers[..., 2].clamp(min=1e-3)
                if multiplier.ndim > 1:
                    mult = multiplier.mean(dim=-1)
                else:
                    mult = multiplier
                mult_flat = mult[voxel_batch_ids]
                scales = scales * depths_flat[:, None, None] * mult_flat[:, None, None]
            rotations = rotations_raw / (rotations_raw.norm(dim=-1, keepdim=True) + 1e-8)
            if voxel_colors is not None and self.add_point_color_residual:
                voxel_colors = voxel_colors.view(-1, 3)
                sh0 = (voxel_colors - 0.5) / 0.28209479177387814
                harmonics = harmonics.clone()
                harmonics[..., 0] = harmonics[..., 0] + sh0[:, None, :]
            if getattr(self, "pred_color", True) is False:
                harmonics = harmonics * self.sh_mask.view(1, 1, 1, -1).to(harmonics)
            opacities = opacity_raw.sigmoid().reshape(-1)

            means = means.reshape(-1, 3)
            scales = scales.reshape(-1, self.scale_dim)
            rotations = rotations.reshape(-1, 4)
            harmonics = harmonics.reshape(-1, 3, self.d_sh)
            voxel_batch_ids = voxel_batch_ids.repeat_interleave(self.gaussian_splits)
            gaussian_voxel_conf = None
            if voxel_depth_conf is not None:
                gaussian_voxel_conf = voxel_depth_conf.reshape(-1).repeat_interleave(self.gaussian_splits)

            means, scales, rotations, harmonics, opacities, gaussian_voxel_conf = self._pack_voxel_gaussians(
                means=means,
                scales=scales,
                rotations=rotations,
                harmonics=harmonics,
                opacities=opacities,
                voxel_conf=gaussian_voxel_conf,
                batch_ids=voxel_batch_ids,
                batch_size=batch_size,
            )
            if opacities.ndim == 3 and opacities.shape[-1] == 1:
                opacities = opacities.squeeze(-1)
            if gaussian_voxel_conf is not None and gaussian_voxel_conf.ndim == 3 and gaussian_voxel_conf.shape[-1] == 1:
                gaussian_voxel_conf = gaussian_voxel_conf.squeeze(-1)

        output["gaussians"] = Gaussians(
            means=means,
            scales=scales,
            rotations=rotations,
            harmonics=harmonics,
            opacities=opacities,
        )
        output["gaussian_voxel_depth_conf"] = gaussian_voxel_conf
        return output
