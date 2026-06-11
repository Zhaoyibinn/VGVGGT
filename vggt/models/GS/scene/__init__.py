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

import os
import random
import json
import numpy as np
import torch
from pathlib import Path
import sys
import trimesh
GS_DIR = Path(__file__).resolve().parents[1]
GS_DIR_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(GS_DIR))
sys.path.append(str(GS_DIR_DIR))


# from GS.utils.system_utils import searchForMaxIteration
# from GS.scene.dataset_readers import sceneLoadTypeCallbacks
from GS.scene.gaussian_model import GaussianModel
import open3d as o3d
# Simple helper to convert extrinsic XYZ Euler angles to a rotation matrix.
def euler_xyz_to_matrix(euler_rad: np.ndarray) -> np.ndarray:
    euler_rad = np.asarray(euler_rad, dtype=np.float32).ravel()
    if euler_rad.size != 3:
        raise ValueError("Expected three Euler angles (roll, pitch, yaw).")
    roll, pitch, yaw = euler_rad
    sx, cx = np.sin(roll), np.cos(roll)
    sy, cy = np.sin(pitch), np.cos(pitch)
    sz, cz = np.sin(yaw), np.cos(yaw)

    rot_x = np.array([[1, 0, 0],
                      [0, cx, -sx],
                      [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy],
                      [0, 1, 0],
                      [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0],
                      [sz, cz, 0],
                      [0, 0, 1]], dtype=np.float32)

    return rot_z @ rot_y @ rot_x


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion conversion failed due to zero norm.")
    return quat / norm


def quaternion_left_multiply(quat_wxyz: torch.Tensor, batch_wxyz: torch.Tensor) -> torch.Tensor:
    if batch_wxyz.shape[-1] != 4:
        raise ValueError("Quaternion batch must have shape (*, 4).")
    quat = quat_wxyz.view(1, 4)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    bw, bx, by, bz = batch_wxyz[:, 0], batch_wxyz[:, 1], batch_wxyz[:, 2], batch_wxyz[:, 3]
    out_w = qw * bw - qx * bx - qy * by - qz * bz
    out_x = qw * bx + qx * bw + qy * bz - qz * by
    out_y = qw * by - qx * bz + qy * bw + qz * bx
    out_z = qw * bz + qx * by - qy * bx + qz * bw
    out = torch.stack((out_w, out_x, out_y, out_z), dim=-1)
    norm = torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(1e-12)
    return out / norm


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).ravel()
    if quaternion.size != 4:
        raise ValueError("Quaternion must have four components (x, y, z, w).")
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


def quaternion_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).ravel()
    if quaternion.size != 4:
        raise ValueError("Quaternion must have four components (x, y, z, w).")
    x, y, z, w = quaternion
    return np.array([w, x, y, z], dtype=np.float32)
# from GS.arguments import ModelParams
# from GS.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    # def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
    def __init__(self, gaussians : GaussianModel, GS_path,extra_T_path):
        """b
        :param path: Path to colmap scene main folder.
        """
        # self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))

        # self.train_cameras = {}
        # self.test_cameras = {}

        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     assert False, "Could not recognize scene type!"

        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #         dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        # if self.loaded_iter:
        self.gaussians.load_ply(GS_path)
        extra_pose = np.loadtxt(extra_T_path)
        translation = np.asarray(extra_pose[0], dtype=np.float32)
        euler_deg = np.asarray(extra_pose[1], dtype=np.float32)
        if extra_pose.shape[0] > 2:
            scale_vals = np.asarray(extra_pose[2], dtype=np.float32).ravel()
            scale = float(scale_vals[0]) if scale_vals.size else 1.0
        else:
            scale = 1.0
        rotation_rad = np.deg2rad(euler_deg)

        def apply_global_rotation(rotation_quat: np.ndarray, gaussian_rotation: torch.Tensor):
            device = gaussians._rotation.device
            quat_tensor = torch.from_numpy(rotation_quat).to(device)
            with torch.no_grad():
                updated = quaternion_left_multiply(quat_tensor, gaussian_rotation)
            return updated
        
        def transform_gaussian_xyz(rotation_matrix: np.ndarray, translation: np.ndarray, scale: float,gaussian_xyz):
            device = gaussian_xyz.device
            rot = torch.from_numpy(np.asarray(rotation_matrix, dtype=np.float32)).to(device)
            trans = torch.from_numpy(np.asarray(translation, dtype=np.float32)).to(device)
            scale_val = float(scale)
            with torch.no_grad():
                xyz = torch.matmul(gaussian_xyz, rot.t())
                xyz = xyz + trans
                if not np.isclose(scale_val, 1.0):
                    xyz = xyz * scale_val
            return xyz
        rotation_matrix = euler_xyz_to_matrix(rotation_rad)
        rotation_quat = rotation_matrix_to_quaternion(rotation_matrix)
        rotated_gaussian_rotation = apply_global_rotation(rotation_quat,self.gaussians._rotation)
        rotated_gaussian_xyz = transform_gaussian_xyz(rotation_matrix, translation, scale,self.gaussians._xyz)
        rotated_gaussian_scale = self.gaussians._scaling + np.log(scale)

        device = rotated_gaussian_xyz.device
        y_up_to_z_up = torch.tensor(
            trimesh.transformations.euler_matrix(np.pi / 2, 0, 0),
            dtype=torch.float32,
            device=device,
        )
        y_up_quat = rotation_matrix_to_quaternion(y_up_to_z_up[:3, :3].cpu().numpy())

        rotated_gaussian_xyz_pybullet = torch.matmul(rotated_gaussian_xyz, y_up_to_z_up[:3, :3].T)
        rotated_gaussian_rotation_pybullet = apply_global_rotation(y_up_quat, rotated_gaussian_rotation)
        rotated_gaussian_scale_pybullet = rotated_gaussian_scale

        with torch.no_grad():
            self.gaussians._xyz.copy_(rotated_gaussian_xyz_pybullet)
            self.gaussians._rotation.copy_(rotated_gaussian_rotation_pybullet)
            self.gaussians._scaling.copy_(rotated_gaussian_scale_pybullet)

        self._canonical_xyz = self.gaussians._xyz.detach().clone()
        self._canonical_rotation = self.gaussians._rotation.detach().clone()

        # with torch.no_grad():
        #     self.gaussians._xyz.copy_(rotated_gaussian_xyz)
        #     self.gaussians._rotation.copy_(rotated_gaussian_rotation)
        #     self.gaussians._scaling.copy_(rotated_gaussian_scale)

        # self.gaussians.save_ply("test/test.ply")

        
        
        
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def set_pose(self, position, quaternion_xyzw):
        if not hasattr(self, "_canonical_xyz") or self._canonical_xyz is None:
            self._canonical_xyz = self.gaussians._xyz.detach().clone()
        if not hasattr(self, "_canonical_rotation") or self._canonical_rotation is None:
            self._canonical_rotation = self.gaussians._rotation.detach().clone()

        position = np.asarray(position, dtype=np.float32)
        quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float32)
        rot_matrix = quaternion_xyzw_to_matrix(quaternion_xyzw)
        quat_wxyz = quaternion_xyzw_to_wxyz(quaternion_xyzw)

        device = self.gaussians._xyz.device
        rot = torch.from_numpy(rot_matrix).to(device)
        trans = torch.from_numpy(position).to(device)
        base_xyz = self._canonical_xyz.to(device)
        base_rot = self._canonical_rotation.to(device)

        with torch.no_grad():
            updated_xyz = torch.matmul(base_xyz, rot.t()) + trans
            self.gaussians._xyz.copy_(updated_xyz)

            quat_tensor = torch.from_numpy(quat_wxyz).to(device)
            updated_rot = quaternion_left_multiply(quat_tensor, base_rot)
            self.gaussians._rotation.copy_(updated_rot)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]



