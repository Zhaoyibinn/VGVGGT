
import os
import sys
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from pytorch3d.ops import knn_points, knn_gather

sys.path.append(os.path.join((os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from .blocks import ZeroConvBlock, DownBlock
from .tools.voxel_utils import get_vox_indices, get_vox_centers_from_points, get_vox_centers_from_indices
from ptv3.point_transformer import PointTransformerV3
from tqdm import tqdm




class BackEnd(nn.Module):
    def __init__(self, hash_base=1024, in_dim=1024, out_dim=256, 
                 k_neighbors=16, depth=48):
        super(BackEnd, self).__init__()
        self.base = hash_base
        self.out_dim = out_dim
        self.aligner = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.GELU(),
            nn.Linear(in_dim//2, out_dim),
            nn.GELU()
        )

        self.point_transformer = PointTransformerV3()
        self.k_neighbors = k_neighbors
        self.downsample = DownBlock(in_channels=out_dim, mid_channels=out_dim, out_channels=out_dim)
        self.zero_conv = ZeroConvBlock(in_channels=out_dim, mid_channels=out_dim, out_channels=out_dim)
        self.gate_scale = nn.Parameter(torch.ones(1))
        
        # added to project features back to the original DINO width
        self.out_aligner = nn.Linear(out_dim, in_dim) if out_dim != in_dim else nn.Identity()
        if isinstance(self.out_aligner, nn.Linear):
            nn.init.normal_(self.out_aligner.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.out_aligner.bias)

        # Unused parameters cause DDP backward pass to hang when find_unused_parameters=False
        # self.zero_conv_layers = nn.ModuleList(
        #     [ZeroConvBlock(in_channels=out_dim, mid_channels=out_dim, out_channels=out_dim) for _ in range(depth)]
        # )
        # self.gate_scales = nn.ParameterList(
        #     [nn.Parameter(torch.ones(1)) for _ in range(depth)]
        # )
    #  todo
    # 把除了self.point_transformer的东西都移到bf16
    @staticmethod
    def map_to_args_to_float(args, kwargs):
        args = tuple(
            torch.float32 if isinstance(arg, torch.dtype) else arg
            for arg in args
        )
        kwargs = dict(kwargs)
        for key in kwargs:
            if key == "dtype":
                kwargs[key] = torch.float32
        return args, kwargs
    
    # def to(self, *args, **kwargs):
    #     # aligner / downsample / zero_conv / out_aligner can run in bf16
    #     self.aligner = self.aligner.to(*args, **kwargs) if self.aligner is not None else None
    #     self.zero_conv = self.zero_conv.to(*args, **kwargs) if self.zero_conv is not None else None
        
        
    #     # gate_scale is an nn.Parameter – must follow the same device/dtype as bf16 modules
    #     if self.gate_scale is not None:
    #         self.gate_scale = nn.Parameter(self.gate_scale.to(*args, **kwargs))

    #     # PointTransformerV3 relies on spconv which does not support bf16/fp16 – keep fp32
    #     args, kwargs = self.map_to_args_to_float(args, kwargs)
    #     self.point_transformer = self.point_transformer.to(*args, **kwargs) if self.point_transformer is not None else None
    #     self.downsample = self.downsample.to(*args, **kwargs) if self.downsample is not None else None

    #     self.out_aligner = self.out_aligner.to(*args, **kwargs) if self.out_aligner is not None else None
        
    #     return self

    @torch.no_grad()
    def hash_fn(self, coords):
        '''
        A simple hash function for voxel coordinates
        '''
        b, x, y, z = coords.unbind(dim=1)
        return ((b.long() << 48)
               | (x.long() << 32)
               | (y.long() << 16)
               |  z.long())
    
    
    def mean_by_voxel(self, points, feats, batch_ids, voxel_size, bounding_boxes, colors=None, depth_conf=None):
        '''Compute mean features for each voxel.
        
        Params:
            - points: (N, 3) tensor of point coordinates
            - feats: (N, C) tensor of point features
            - batch_ids: (N,) tensor of batch indices for each point
            - voxel_size: scalar or (3,) tensor defining the size of each voxel
            - bounding_boxes: (B, 2, 3) tensor of min and max coordinates for each batch
            - colors: (N, C) tensor of point colors (optional)
            - depth_conf: (N, C) tensor of point depth confidence (optional)
        
        Returns:
            - voxel_feats: (M, C) tensor of mean features for each voxel
            - info: dict containing 'unique_indices' which are the voxel indices corresponding to the mean features
        
        '''
        ori_device = feats.device 
        # if feats.shape[0]>1000000000:
        #     device = torch.device('cpu')
        # else:
        device = ori_device
        points = points.to(device)
        feats = feats.to(device)
        batch_ids = batch_ids.to(device)
        bounding_boxes = bounding_boxes.to(device)
        colors = colors.to(device) if colors is not None else None
        depth_conf = depth_conf.to(device) if depth_conf is not None else None
        voxel_indices = get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True)
        voxel_hash = self.hash_fn(voxel_indices) 
        unique_hash, inverse_id = torch.unique(voxel_hash, return_inverse=True) # inverse_id表示这个点属于哪个体素
        
        voxel_feats = scatter_mean(feats, inverse_id, dim=0).to(ori_device)
        voxel_colors = scatter_mean(colors, inverse_id, dim=0).to(ori_device) if colors is not None else None
        voxel_depth_conf = scatter_mean(depth_conf, inverse_id, dim=0).to(ori_device) if depth_conf is not None else None
        voxel_centers = get_vox_centers_from_points(
            points,
            depth_conf=depth_conf,
            inverse_id=inverse_id,
            num_voxels=unique_hash.shape[0],
        ).to(ori_device)

        original_indices = torch.arange(voxel_hash.shape[0], device=voxel_hash.device)
        min_original_indices_per_unique_id = torch.full((unique_hash.shape[0],),
                                                voxel_hash.shape[0],
                                                dtype=torch.long,
                                                device=device)
        
        first_occurrence_original_indices = torch.scatter_reduce(
            min_original_indices_per_unique_id,
            0,
            inverse_id,
            original_indices,
            reduce="amin",
            include_self=False
        )

        unique_voxel_indices = voxel_indices[first_occurrence_original_indices].to(ori_device)
        

        # Release intermediate tensors to free VRAM (inference only)
        if not self.training:
            del voxel_hash, unique_hash, inverse_id, original_indices
            del min_original_indices_per_unique_id, first_occurrence_original_indices

        info = {
            'unique_indices': unique_voxel_indices,
            'voxel_centers': voxel_centers,
            'colors': voxel_colors,
            'depth_conf': voxel_depth_conf,
        }    

        return voxel_feats, info

    
    def voxel_to_point_interpolation(self, point_out, pts, chunk_size=50000):
        """Interpolate point/voxel features back to exact continuous points via batched chunked KNN."""
        Bs = pts.shape[0] if len(pts.shape) == 3 else (pts.shape[0] // pts.shape[1] if hasattr(pts, 'shape') and len(pts.shape) == 2 else 1)
        if len(pts.shape) == 2:
            Bs = point_out.batch.max().item() + 1
            N = pts.shape[0] // Bs
        else:
            Bs, N, _ = pts.shape

        pts_feat_from_voxel = point_out.feat      # (V, C_out)
        pts_coord_from_voxel = point_out.coord    # (V, 3)
        pts_batch_from_voxel = point_out.batch    # (V,)

        original_pts = pts.view(Bs, N, 3)         # (Bs, N, 3)

        voxel_coords_split = [pts_coord_from_voxel[pts_batch_from_voxel == b] for b in range(Bs)]
        voxel_feats_split = [pts_feat_from_voxel[pts_batch_from_voxel == b] for b in range(Bs)]
        voxel_lens = torch.tensor([v.shape[0] for v in voxel_coords_split], device=pts.device)

        max_T = int(voxel_lens.max().item())
        pad_3 = (0, 0, 0, max_T)
        pad_C = (0, 0, 0, max_T)

        voxel_coords_padded = torch.stack(
            [F.pad(v, pad_3, value=0.)[:max_T] for v in voxel_coords_split]
        )  # (Bs, max_T, 3)
        voxel_feats_padded = torch.stack(
            [F.pad(v, pad_C, value=0.)[:max_T] for v in voxel_feats_split]
        )  # (Bs, max_T, C_out)

        K_interp = self.k_neighbors
        knn = knn_points(
            original_pts,
            voxel_coords_padded,
            lengths2=voxel_lens,
            K=K_interp,
        )
        dists = knn.dists  # (Bs, N, K)
        idx = knn.idx      # (Bs, N, K)

        interpolated_feats_chunks = []
        num_chunks = (N + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)

            idx_chunk = idx[:, start_idx:end_idx, :]      # (Bs, chunk_size, K)
            dists_chunk = dists[:, start_idx:end_idx, :]  # (Bs, chunk_size, K)

            gathered_feats_chunk = knn_gather(voxel_feats_padded, idx_chunk)  # (Bs, chunk_size, K, C_out)
            weights_chunk = dists_chunk.clamp(min=1e-8).reciprocal()
            weights_chunk = (weights_chunk / weights_chunk.sum(dim=-1, keepdim=True)).to(pts.dtype)

            gathered_feats_chunk = gathered_feats_chunk.to(pts.dtype)
            interpolated_chunk = (gathered_feats_chunk * weights_chunk.unsqueeze(-1)).sum(dim=-2)

            if not self.training:
                del gathered_feats_chunk, weights_chunk
                torch.cuda.empty_cache()

            interpolated_feats_chunks.append(interpolated_chunk)

        interpolated_feats = torch.cat(interpolated_feats_chunks, dim=1)  # (Bs, N, C_out)
        return interpolated_feats


    def forward(self, pts, feats, voxel_sizes, chunk_size=50000, return_voxel_details=False, colors=None, depth_conf=None, skip_interpolation=False):
        '''
        Forward pass for the back-end processing.
        
        Params:
            - pts: (Bs, N, 3) tensor of point coordinates
            - feats: (Bs, N, C) tensor of point features
            - voxel_sizes: list of voxel sizes
            - chunk_size: int, number of points to process in each chunk for interpolation
            - colors: (Bs, N, C) tensor of point colors (optional)
            - depth_conf: (Bs, N, C) tensor of point depth confidence (optional)
            - skip_interpolation: bool, if True, skips the expensive point interpolation
        '''
        # assert False

        if isinstance(feats, list):
            if not skip_interpolation:
                raise ValueError("List backend inputs are only supported when skip_interpolation=True")
            Bs = len(feats)
            C = feats[0].shape[-1] if Bs > 0 else 0
        else:
            Bs, C = feats.shape[0], feats.shape[-1]
        # Voxel hashing and sparse conv are both precision-sensitive. Always run the
        # backend core in fp32 regardless of outer autocast state.


        if not isinstance(feats, list) and len(feats.shape) != 3:
            feats = feats.reshape(Bs, -1, C)
            pts = pts.reshape(Bs, -1, 3)
            if colors is not None:
                colors = colors.reshape(Bs, -1, colors.shape[-1])
            if depth_conf is not None:
                depth_conf = depth_conf.reshape(Bs, -1, depth_conf.shape[-1])

        # autocast_ctx = torch.amp.autocast("cuda", enabled=False) if pts.is_cuda else contextlib.nullcontext()
        # with autocast_ctx:
        level_feats = []
        voxel_details = []

        for i, voxel_size in enumerate(voxel_sizes):
            interpolated_feats_per_batch = []
            voxel_feat_per_batch = []
            voxel_center_per_batch = []
            voxel_batch_ids_per_batch = []
            voxel_color_per_batch = []
            voxel_depth_conf_per_batch = []

            for batch_idx in range(Bs):
                pts_b = pts[batch_idx]      # (N, 3)
                feats_b = feats[batch_idx]  # (N, C)
                colors_b = colors[batch_idx] if colors is not None else None
                depth_conf_b = depth_conf[batch_idx] if depth_conf is not None else None

                if pts_b.numel() == 0:
                    interpolated_feats_per_batch.append(None)
                    if return_voxel_details:
                        voxel_feat_per_batch.append(feats_b.new_zeros((0, feats_b.shape[-1])))
                        voxel_center_per_batch.append(pts_b.new_zeros((0, 3)))
                        voxel_batch_ids_per_batch.append(torch.zeros((0,), device=pts_b.device, dtype=torch.long))
                        voxel_color_per_batch.append(colors_b.new_zeros((0, colors_b.shape[-1])) if colors_b is not None else None)
                        voxel_depth_conf_per_batch.append(depth_conf_b.new_zeros((0, depth_conf_b.shape[-1])) if depth_conf_b is not None else None)
                    continue

                bounding_boxes_b = torch.zeros((1, 2, 3), device=pts_b.device)
                bounding_boxes_b[:, 0, :] = pts_b.min(dim=0, keepdim=True).values
                bounding_boxes_b[:, 1, :] = pts_b.max(dim=0, keepdim=True).values

                batch_ids_b = torch.zeros(pts_b.shape[0], device=pts_b.device, dtype=torch.long)

                feat_b, info_b = self.mean_by_voxel(
                    pts_b,
                    feats_b,
                    batch_ids_b,
                    voxel_size,
                    bounding_boxes_b,
                    colors=colors_b,
                    depth_conf=depth_conf_b,
                )
                vox_id_b = info_b['unique_indices']
                coord_b = info_b['voxel_centers']

                if False:
                    import numpy as np
                    import open3d as o3d

                    save_root = "test_voxel"
                    os.makedirs(save_root, exist_ok=True)

                    voxel_size_value = voxel_size
                    if torch.is_tensor(voxel_size_value):
                        voxel_size_value = float(voxel_size_value.reshape(-1)[0].item())
                    else:
                        voxel_size_value = float(voxel_size_value)

                    coord_np = coord_b.detach().float().cpu().numpy()
                    color_b = info_b.get('colors', None)
                    if color_b is not None:
                        color_np = color_b.detach().float().cpu().numpy().clip(0.0, 1.0)
                    else:
                        color_np = np.tile(np.array([[0.2, 0.7, 1.0]], dtype=np.float32), (coord_np.shape[0], 1))

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coord_np)
                    pcd.colors = o3d.utility.Vector3dVector(color_np)
                    o3d.io.write_point_cloud(
                        os.path.join(save_root, f"level{i}_batch{batch_idx}_voxel_centers.ply"),
                        pcd,
                        write_ascii=True,
                    )

                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size_value)
                    voxel_mesh = o3d.geometry.TriangleMesh()
                    for voxel in voxel_grid.get_voxels():
                        cube = o3d.geometry.TriangleMesh.create_box(voxel_size_value, voxel_size_value, voxel_size_value)
                        cube.translate(voxel_grid.origin + np.asarray(voxel.grid_index, dtype=np.float64) * voxel_size_value)
                        cube.paint_uniform_color(voxel.color)
                        voxel_mesh += cube

                    if len(voxel_mesh.vertices) > 0:
                        voxel_mesh.compute_vertex_normals()
                        o3d.io.write_triangle_mesh(
                            os.path.join(save_root, f"level{i}_batch{batch_idx}_voxels.ply"),
                            voxel_mesh,
                            write_ascii=True,
                        )

                data_dict_b = {
                    'feat': feat_b,
                    'grid_coord': vox_id_b[:, 1:],
                    'coord': coord_b,
                    'batch': vox_id_b[:, 0],
                }

                if not skip_interpolation:
                    point_out_b = self.point_transformer(data_dict_b)
                    interpolated_b = self.voxel_to_point_interpolation(
                        point_out_b,
                        pts_b.unsqueeze(0),
                        chunk_size,
                    )
                    interpolated_feats_per_batch.append(interpolated_b)
                    out_feat = point_out_b.feat
                    out_coord = point_out_b.coord
                else:
                    interpolated_feats_per_batch.append(None)
                    out_feat = feat_b
                    out_coord = coord_b

                if return_voxel_details:
                    voxel_feat_per_batch.append(out_feat)
                    voxel_center_per_batch.append(out_coord)
                    voxel_batch_ids_per_batch.append(
                        torch.full(
                            (out_feat.shape[0],),
                            batch_idx,
                            device=out_feat.device,
                            dtype=torch.long,
                        )
                    )
                    voxel_color_per_batch.append(info_b.get('colors', None))
                    voxel_depth_conf_per_batch.append(info_b.get('depth_conf', None))

            if not skip_interpolation:
                interpolated_feats = torch.cat(interpolated_feats_per_batch, dim=0)
                level_feats.append(interpolated_feats)
            else:
                level_feats.append(None)

            if return_voxel_details:
                voxel_details.append(
                    {
                        "voxel_feat": voxel_feat_per_batch,
                        "voxel_centers": voxel_center_per_batch,
                        "voxel_batch_ids": voxel_batch_ids_per_batch,
                        "voxel_colors": voxel_color_per_batch,
                        "voxel_depth_conf": voxel_depth_conf_per_batch,
                        "voxel_size": voxel_size,
                    }
                )
        
        if return_voxel_details:
            return level_feats, voxel_details
        return level_feats
