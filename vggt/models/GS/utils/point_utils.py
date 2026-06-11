import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    device = depthmap.device
    with torch.amp.autocast("cuda", enabled=False):
        c2w = torch.linalg.inv(view.world_view_transform.T.to(device=device, dtype=torch.float32))
        W, H = view.image_width, view.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]], device=device, dtype=torch.float32).T
        projection_matrix = c2w.T @ view.full_proj_transform.to(device=device, dtype=torch.float32)
        intrins = ((projection_matrix @ ndc2pix)[:3, :3].T).to(dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing='xy'
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ torch.linalg.inv(intrins).T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        world_points = depthmap.to(device=device, dtype=torch.float32).reshape(-1, 1) * rays_d + rays_o
    return world_points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output