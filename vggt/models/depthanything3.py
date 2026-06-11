import copy
import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file
from wcmatch import fnmatch

from vggt.models.depth_anything_3.cfg import create_object
from vggt.utils.specs import Gaussians
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

from vggt.utils.gsply_helpers import save_gaussian_ply

from vggt.utils.geometry import map_pdf_to_opacity
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.sh_helpers import RGB2SH

from vggt.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode,render_3dgs

from vggt.models.GS.utils.build_camera import build_gs_camera  
from vggt.models.GS.gaussian_renderer import GaussianModel,render, network_gui


class DepthAnything3(nn.Module):
    DEFAULT_PRETRAIN_DIRS = [
        "da3_streaming/weights_gaint_large_1.1",
        "/home/zhaoyibin/3DRE/MVS/Depth-Anything-3/da3_streaming/weights_gaint_large_1.1",
    ]
    _GS_OPS_CACHE = None
    _GLOB_FLAGS = (
        fnmatch.CASE
        | fnmatch.DOTMATCH
        | fnmatch.EXTMATCH
        | fnmatch.SPLIT
    )

    def __init__(
        self,
        model_name: str = "da3nested-giant-large",
        infer_gs: bool = False,
        gs_from_backend: bool = False,
        use_manual_metric_scaling: bool = False,
        manual_metric_divisor: float = 300.0,
        config: Dict[str, Any] | None = None,
        pretrained_dir: str | None = None,
        config_path: str | None = None,
        pretrained_weight: str | None = None,
        weights_path: str | None = None,
        load_pretrained: bool = True,
        skip_load_module_names: list[str] | None = None,
        gs_mode: None = None,
        # use_ray_pose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.infer_gs = infer_gs
        self.gs_from_backend = gs_from_backend
        self.use_manual_metric_scaling = use_manual_metric_scaling
        self.manual_metric_divisor = float(manual_metric_divisor)
        self.gs_mode = gs_mode
        self.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        if config is None:
            if config_path is None:
                raise ValueError("DepthAnything3 requires `config` or `config_path`; no implicit fallback to pretrained_dir/config.json")
            resolved_config_path = config_path
            with open(resolved_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            config = loaded.get("config", loaded)

        config = self._rewrite_object_paths(copy.deepcopy(config))
        self.model = create_object(OmegaConf.create(config))

        if self.gs_from_backend:
            # gs_head = getattr(self.model, "gs_head", None)
            # if isinstance(gs_head, nn.Module):
            logging.info("`gs_from_backend=True`: freeze unused `model.gs_head` and `model.gs_adapter` parameters for DDP.")
            gs_head = getattr(self.model, "gs_head", getattr(getattr(self.model, "da3", None), "gs_head", None))
            if gs_head is not None:
                for param in gs_head.parameters():
                    param.requires_grad = False
            # gs_adapter = getattr(self.model, "gs_adapter", None)
            # if isinstance(gs_adapter, nn.Module):
            gs_adapter = getattr(self.model, "gs_adapter", getattr(getattr(self.model, "da3", None), "gs_adapter", None))
            if gs_adapter is not None:
                for param in gs_adapter.parameters():
                    param.requires_grad = False

        if load_pretrained:
            resolved_weights_path = pretrained_weight or weights_path
            if resolved_weights_path is None:
                resolved_pretrained_dir = self._resolve_pretrained_dir(pretrained_dir)
                resolved_weights_path = os.path.join(resolved_pretrained_dir, "model.safetensors")
            
            if resolved_weights_path.endswith('.safetensors'):
                state_dict = load_file(resolved_weights_path)
            else:
                state_dict = torch.load(resolved_weights_path, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            
            if skip_load_module_names:
                self._assert_skip_module_patterns_exist(skip_load_module_names)
                state_dict, dropped_keys = self._filter_state_dict_by_module_patterns(
                    state_dict,
                    skip_load_module_names,
                )
                logging.info(
                    "Skip loading %d tensors for module patterns: %s",
                    dropped_keys,
                    skip_load_module_names,
                )
            state_dict, mismatched_keys = self._filter_state_dict_by_shape(state_dict)
            if mismatched_keys:
                logging.warning(
                    "Skip loading %d tensors due to shape mismatch: %s",
                    len(mismatched_keys),
                    mismatched_keys,
                )
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.info("Missing keys when loading pretrained weights: %s", missing_keys)
            if unexpected_keys:
                logging.info("Unexpected keys when loading pretrained weights: %s", unexpected_keys)

    def to(self, *args, **kwargs):
        # TODO: this won't work if the module is inside another module
        self.model = self.model.to(*args, **kwargs)
        self._imagenet_mean = self._imagenet_mean.to(*args, **kwargs)
        self._imagenet_std = self._imagenet_std.to(*args, **kwargs)
        return self

    def _assert_skip_module_patterns_exist(self, patterns: list[str]) -> None:
        module_names = [name for name, _ in self.named_modules()]
        not_found = [
            p
            for p in patterns
            if not any(fnmatch.fnmatch(name, p, flags=self._GLOB_FLAGS) for name in module_names)
        ]
        assert not not_found, f"These skip_load_module_names patterns matched no modules: {not_found}"

    def _filter_state_dict_by_module_patterns(
        self,
        state_dict: Dict[str, torch.Tensor],
        patterns: list[str],
    ):
        filtered = {}
        dropped = 0
        for key, value in state_dict.items():
            should_drop = False
            for pattern in patterns:
                if fnmatch.fnmatch(key, pattern, flags=self._GLOB_FLAGS) or fnmatch.fnmatch(
                    key,
                    f"{pattern}.*",
                    flags=self._GLOB_FLAGS,
                ):
                    should_drop = True
                    break

            if should_drop:
                dropped += 1
                continue

            filtered[key] = value

        return filtered, dropped

    def _filter_state_dict_by_shape(self, state_dict: Dict[str, torch.Tensor]):
        current_state_dict = self.state_dict()
        filtered = {}
        mismatched_keys = []

        for key, value in state_dict.items():
            current_value = current_state_dict.get(key)
            if current_value is not None and torch.is_tensor(value) and torch.is_tensor(current_value):
                if value.shape != current_value.shape:
                    mismatched_keys.append(
                        f"{key}: checkpoint {tuple(value.shape)} != model {tuple(current_value.shape)}"
                    )
                    continue

            filtered[key] = value

        return filtered, mismatched_keys

    def _resolve_pretrained_dir(self, pretrained_dir: str | None) -> str:
        if pretrained_dir is not None:
            return pretrained_dir

        for candidate in self.DEFAULT_PRETRAIN_DIRS:
            if os.path.exists(candidate):
                return candidate

        return self.DEFAULT_PRETRAIN_DIRS[0]

    def _rewrite_object_paths(self, cfg: Any) -> Any:
        if isinstance(cfg, dict):
            out = {}
            for key, value in cfg.items():
                if key == "__object__" and isinstance(value, dict) and "path" in value:
                    value = copy.deepcopy(value)
                    path = value["path"]
                    if isinstance(path, str) and path.startswith("depth_anything_3."):
                        value["path"] = f"vggt.models.{path}"
                out[key] = self._rewrite_object_paths(value)
            return out

        if isinstance(cfg, list):
            return [self._rewrite_object_paths(item) for item in cfg]

        return cfg

    def _maybe_normalize_images(self, images: torch.Tensor, normalize_images: bool = True) -> torch.Tensor:
        if not normalize_images:
            return images

        img_min = images.amin().item()
        img_max = images.amax().item()
        if img_min >= -1e-3 and img_max <= 1.0 + 1e-3:
            images = (images - self._imagenet_mean) / self._imagenet_std
        return images

    @staticmethod
    def _infer_sh_degree(gs_world) -> int:
        harmonics = getattr(gs_world, "harmonics", None)
        if harmonics is None or harmonics.shape[-2] <= 0:
            return 0

        coeff_count = int(harmonics.shape[-1])
        sh_degree = int(coeff_count**0.5) - 1
        return max(sh_degree, 0)

    def _compute_manual_metric_scales(self, depth: torch.Tensor, intrinsics: torch.Tensor):
        if self.manual_metric_divisor <= 0:
            raise ValueError(f"manual_metric_divisor must be > 0, got {self.manual_metric_divisor}")

        focal = 0.5 * (intrinsics[..., 0, 0] + intrinsics[..., 1, 1])
        if depth.ndim == 5:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None, None]
        elif depth.ndim == 4:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None]
        else:
            raise ValueError(f"Unexpected depth ndim: {depth.ndim}")

        scene_scale = (focal / self.manual_metric_divisor).mean(dim=1)
        return depth_scale, scene_scale

    @staticmethod
    def _scale_extrinsics_translation(extrinsics: torch.Tensor, scene_scale: torch.Tensor) -> torch.Tensor:
        scaled_t = extrinsics[..., :3, 3] * scene_scale[:, None, None]
        top = torch.cat([extrinsics[..., :3, :3], scaled_t.unsqueeze(-1)], dim=-1)
        if extrinsics.shape[-2] == 4:
            bottom = extrinsics[..., 3:4, :]
            return torch.cat([top, bottom], dim=-2)
        return top

    @staticmethod
    def _scale_gaussians(gs_world, scene_scale: torch.Tensor):
        if gs_world is None:
            return gs_world

        scale = scene_scale[:, None, None]
        if hasattr(gs_world, "means") and torch.is_tensor(gs_world.means):
            means = gs_world.means * scale
            scales = gs_world.scales * scale if torch.is_tensor(getattr(gs_world, "scales", None)) else gs_world.scales
            return Gaussians(
                means=means,
                scales=scales,
                rotations=gs_world.rotations,
                harmonics=gs_world.harmonics,
                opacities=gs_world.opacities,
            )
        if isinstance(gs_world, dict):
            out = dict(gs_world)
            if torch.is_tensor(gs_world.get("means", None)):
                out["means"] = gs_world["means"] * scale
            if torch.is_tensor(gs_world.get("scales", None)):
                out["scales"] = gs_world["scales"] * scale
            return out
        return gs_world

    def _build_predictions_from_output(
        self,
        output,
        images: torch.Tensor,
        image_hw: tuple[int, int],
        render_gs: bool,
        sh_degree: int | None,
    ) -> dict[str, Any]:
        depth = output.depth
        if depth.ndim == 4:
            depth = depth.unsqueeze(-1)

        depth_conf = output.get("depth_conf", None)
        # voxel_depth_conf = output.get("voxel_depth_conf", None)
        gaussian_voxel_depth_conf = output.get("gaussian_voxel_depth_conf", None)
        highres_backend_voxel_point_ratio = output.get("highres_backend_voxel_point_ratio", None)
        if depth_conf is None:
            depth_conf = torch.ones_like(depth[..., 0])

        extrinsics = output.extrinsics
        intrinsics = output.intrinsics

        if self.use_manual_metric_scaling:
            depth_scale, scene_scale = self._compute_manual_metric_scales(depth, intrinsics)
            depth = depth * depth_scale
            extrinsics = self._scale_extrinsics_translation(extrinsics, scene_scale)

        if extrinsics.shape[-2:] == (4, 4):
            extrinsics = extrinsics[..., :3, :]

        pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_hw)

        predictions = {
            "pose_enc": pose_enc,
            "pose_enc_list": [pose_enc],
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth": depth,
            "depth_conf": depth_conf,
            # "voxel_depth_conf": voxel_depth_conf,
            "gaussian_voxel_depth_conf": gaussian_voxel_depth_conf,
        }

        if highres_backend_voxel_point_ratio is not None:
            predictions["highres_backend_voxel_point_ratio"] = highres_backend_voxel_point_ratio

        if "gaussians" not in output:
            return predictions

        gs_world = output.gaussians
        if self.use_manual_metric_scaling:
            gs_world = self._scale_gaussians(gs_world, scene_scale)
        gs_world = self._normalize_gaussians_opacity_shape(gs_world)
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

        if not render_gs:
            return predictions

        with torch.cuda.amp.autocast(enabled=False):
            last_row = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
            last_row[..., 0, 3] = 1.0
            extrinsics_h = torch.cat([extrinsics, last_row], dim=-2)

            cam_list_all = build_gs_camera(
                K=intrinsics,
                ext=extrinsics_h,
                height=image_hw[0],
                width=image_hw[1],
                data_device=images.device,
            )
            gs_background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=images.device)
            gs_pipe = SimpleNamespace(
                convert_SHs_python=False,
                compute_cov3D_python=False,
                depth_ratio=0.0,
                kernel_size=0.0,
                require_depth=True,
                debug=False,
            )
            if sh_degree is None:
                sh_degree = self._infer_sh_degree(gs_world)

            render_pkgs = []
            for batch_idx in range(images.shape[0]):
                render_pkgs_batch = []
                for view_idx in range(images.shape[1]):
                    render_pkg = render(
                        cam_list_all[batch_idx][view_idx],
                        gs_world,
                        gs_pipe,
                        gs_background,
                        batch_idx=batch_idx,
                        sh_degree=sh_degree,
                        gs_mode=self.gs_mode
                    )
                    render_pkgs_batch.append(render_pkg)
                render_pkgs.append(render_pkgs_batch)

            predictions["GS_render_pkgs"] = render_pkgs

        return predictions

    @staticmethod
    def _normalize_gaussians_opacity_shape(gs_world):
        if gs_world is None:
            return gs_world
        opacities = getattr(gs_world, "opacities", None)
        if not torch.is_tensor(opacities):
            return gs_world

        if opacities.ndim >= 3 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        elif opacities.ndim >= 3:
            opacities = opacities.reshape(opacities.shape[0], opacities.shape[1], -1)[..., 0]

        if hasattr(gs_world, "opacities"):
            gs_world.opacities = opacities
        elif isinstance(gs_world, dict):
            gs_world["opacities"] = opacities
        return gs_world



    @classmethod
    def _prepare_gs_legacy_import_paths(cls) -> None:
        gs_root = Path(__file__).resolve().parent / "GS"
        models_root = gs_root.parent
        for path in (gs_root, models_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)

    @classmethod
    def _get_gs_ops(cls):
        if cls._GS_OPS_CACHE is not None:
            return cls._GS_OPS_CACHE

        cls._prepare_gs_legacy_import_paths()
        render_fn = import_module("vggt.models.GS.gaussian_renderer").render
        build_camera_fn = import_module("vggt.models.GS.utils.build_camera").build_gs_camera
        cls._GS_OPS_CACHE = (render_fn, build_camera_fn)
        return cls._GS_OPS_CACHE

    def forward(self, images: torch.Tensor, forward_dict: dict = None,**kwargs):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        _, _, _, H, W = images.shape
        ori_images = copy.deepcopy(images)

        normalize_images = kwargs.get("normalize_images", True)
        images = self._maybe_normalize_images(images, normalize_images=normalize_images)

        infer_gs = kwargs.get("infer_gs", self.infer_gs)
        gs_from_backend = kwargs.get("gs_from_backend", self.gs_from_backend)
        use_ray_pose = kwargs.get("use_ray_pose", False)
        ref_view_strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        extrinsics = kwargs.get("extrinsics", None)
        intrinsics = kwargs.get("intrinsics", None)
        render_gs = kwargs.get("render_gs", True)
        sh_degree = kwargs.get("sh_degree", None)
        backend_token_add = forward_dict.get("backend_token_add", True) if forward_dict is not None else True   
        output = self.model(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            rgb_images=ori_images,
            infer_gs=infer_gs,
            gs_from_backend=gs_from_backend,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            backend_token_add = backend_token_add
        )
        predictions = self._build_predictions_from_output(
            output,
            images=images,
            image_hw=(H, W),
            render_gs=render_gs,
            sh_degree=sh_degree,
        )

        raw_stage_outputs = output.get("stage_outputs", None)
        if raw_stage_outputs:
            stage_predictions = []
            for stage_idx, stage_output in enumerate(raw_stage_outputs):
                stage_predictions.append(
                    self._build_predictions_from_output(
                        stage_output,
                        images=images,
                        image_hw=(H, W),
                        render_gs=render_gs and stage_idx == len(raw_stage_outputs) - 1,
                        sh_degree=sh_degree,
                    )
                )
            predictions["baseline"] = stage_predictions[0]
            predictions["refined"] = stage_predictions[-1]
            predictions["stages"] = stage_predictions

        return predictions
