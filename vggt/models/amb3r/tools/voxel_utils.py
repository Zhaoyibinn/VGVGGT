import torch

def get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True):
    '''Compute voxel indices for a batch of points.
    
    Args:
        points (torch.Tensor): A batch of points with shape (N, 3).
        batch_ids (torch.Tensor): A tensor of batch IDs for each point with shape (N,).
        voxel_size (torch.Tensor | float): The size of each voxel.
        bounding_boxes (torch.Tensor): The bounding boxes for each batch with shape (B, 2, 3).
        shift (bool): Whether to shift the voxel indices by half a voxel.
        cat_batch_ids (bool): Whether to concatenate the batch IDs to the voxel indices.

    Returns:
        voxel_indices (torch.Tensor): The computed voxel indices with shape (N, 4) if cat_batch_ids is True, else (N, 3).
    '''
    
    bb_mins = bounding_boxes[batch_ids][:, 0]  # select min box per point
    voxel_size_b = voxel_size[batch_ids] if isinstance(voxel_size, torch.Tensor) else voxel_size
    if isinstance(voxel_size_b, torch.Tensor) and voxel_size_b.dim() == 1:
        voxel_size_b = voxel_size_b.unsqueeze(-1)
        
    if shift:
        bb_mins = bb_mins - 0.5 * voxel_size_b
    
    voxel_indices = torch.floor((points - bb_mins) / voxel_size_b).long()

    if cat_batch_ids:
        voxel_indices = torch.cat([batch_ids.unsqueeze(1), voxel_indices], dim=1)
    
    return voxel_indices

def get_vox_centers_from_indices(voxel_indices, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True):
    '''Compute voxel centers from voxel indices.
    
    Args:
        voxel_indices (torch.Tensor): The voxel indices with shape (N, 4)
        batch_ids (torch.Tensor): A tensor of batch IDs for each voxel with shape (N,).
        voxel_size (torch.Tensor | float): The size of each voxel.
        bounding_boxes (torch.Tensor): The bounding boxes for each batch with shape (B, 2, 3).
        shift (bool): Whether to shift the voxel centers by half a voxel.
        cat_batch_ids (bool): Whether the voxel indices include batch IDs.

    Returns:
        voxel_centers (torch.Tensor): The computed voxel centers with shape (N, 3).
    '''

    bb_mins = bounding_boxes[batch_ids][:, 0]
    voxel_size_b = voxel_size[batch_ids] if isinstance(voxel_size, torch.Tensor) else voxel_size
    if isinstance(voxel_size_b, torch.Tensor) and voxel_size_b.dim() == 1:
        voxel_size_b = voxel_size_b.unsqueeze(-1)

    if shift and cat_batch_ids:
        voxel_centers = voxel_indices[:, 1:] * voxel_size_b + bb_mins
    
    elif (not shift) and cat_batch_ids:
        voxel_centers = voxel_indices[:, 1:] * voxel_size_b + bb_mins + 0.5 * voxel_size_b

    else:
        raise NotImplementedError("Only shift and cat_batch_ids are supported")

    return voxel_centers


def get_vox_centers_from_points(points, depth_conf=None, voxel_indices=None, inverse_id=None, num_voxels=None):
    '''Compute voxel centers from the weighted mean of the points inside each voxel.

    Args:
        points (torch.Tensor): Point coordinates with shape (N, 3).
        depth_conf (torch.Tensor | None): Per-point confidence weights. If the last
            dimension is not 1, the mean over the last dimension is used as the scalar
            weight for each point. If None, falls back to uniform weights.
        voxel_indices (torch.Tensor | None): Per-point voxel indices. Required when
            inverse_id is not provided.
        inverse_id (torch.Tensor | None): Mapping from each point to its voxel id.
        num_voxels (int | None): Number of voxels. Required only when inverse_id is
            provided and the tensor is empty.

    Returns:
        voxel_centers (torch.Tensor): Weighted voxel centers with shape (M, 3).
    '''

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if inverse_id is None:
        if voxel_indices is None:
            raise ValueError("voxel_indices must be provided when inverse_id is None")
        _, inverse_id = torch.unique(voxel_indices, dim=0, return_inverse=True)
        num_voxels = int(inverse_id.max().item()) + 1 if inverse_id.numel() > 0 else 0
    elif num_voxels is None:
        num_voxels = int(inverse_id.max().item()) + 1 if inverse_id.numel() > 0 else 0

    if depth_conf is None:
        weights = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    else:
        weights = depth_conf.to(device=points.device, dtype=points.dtype)
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        else:
            weights = weights.reshape(points.shape[0], -1)
        if weights.shape[1] != 1:
            weights = weights.mean(dim=-1, keepdim=True)
        weights = weights.clamp_min(0)

    weighted_point_sums = torch.zeros((num_voxels, 3), device=points.device, dtype=points.dtype)
    weighted_point_sums.index_add_(0, inverse_id, points * weights)

    weight_sums = torch.zeros((num_voxels, 1), device=points.device, dtype=points.dtype)
    weight_sums.index_add_(0, inverse_id, weights)

    point_sums = torch.zeros((num_voxels, 3), device=points.device, dtype=points.dtype)
    point_sums.index_add_(0, inverse_id, points)

    point_counts = torch.zeros((num_voxels, 1), device=points.device, dtype=points.dtype)
    point_counts.index_add_(0, inverse_id, torch.ones_like(weights))

    weighted_centers = weighted_point_sums / weight_sums.clamp_min(1e-12)
    mean_centers = point_sums / point_counts.clamp_min(1)
    voxel_centers = torch.where(weight_sums > 0, weighted_centers, mean_centers)

    return voxel_centers

