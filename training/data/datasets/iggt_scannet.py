# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import *


class IGGTScanNet(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        IGGT_SCANNET_DIR: str = "train_data/iggt/processed_scannetpp_v2",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 1000.0,
        depth_max: float = 30.0,
        pose_format: str = "c2w",
    ):
        """
        Initialize the IGGT ScanNet++ dataset.

        Expected scene layout:
            {IGGT_SCANNET_DIR}/{scene_id}/
                images/frame_000000.jpg
                depth/frame_000000.png
                scene_iphone_metadata.npz

        The metadata npz is expected to contain:
            trajectories: [N, 4, 4]
            intrinsics: [N, 3, 3]
            images: [N] image filenames
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.IGGT_SCANNET_DIR = IGGT_SCANNET_DIR
        self.min_num_images = min_num_images
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.pose_format = pose_format

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        if self.pose_format not in ("c2w", "w2c"):
            raise ValueError(f"Unsupported pose_format: {self.pose_format}")

        logging.info(f"IGGT_SCANNET_DIR is {self.IGGT_SCANNET_DIR}")

        self.scene_infos = self._build_scene_infos()
        self.sequence_list = [scene_info["scene_path"] for scene_info in self.scene_infos]
        self.sequence_list_len = len(self.sequence_list)

        if self.sequence_list_len == 0:
            raise ValueError(f"No valid IGGT ScanNet++ scenes found in {self.IGGT_SCANNET_DIR}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: IGGT ScanNet++ Data size: {self.sequence_list_len}")
        logging.info(f"{status}: IGGT ScanNet++ dataset length: {len(self)}")

    def _build_scene_infos(self):
        scene_infos = []
        for scene_name in sorted(os.listdir(self.IGGT_SCANNET_DIR)):
            scene_path = osp.join(self.IGGT_SCANNET_DIR, scene_name)
            image_dir = osp.join(scene_path, "images")
            depth_dir = osp.join(scene_path, "depth")
            metadata_path = osp.join(scene_path, "scene_iphone_metadata.npz")

            if not (
                osp.isdir(scene_path)
                and osp.isdir(image_dir)
                and osp.isdir(depth_dir)
                and osp.isfile(metadata_path)
            ):
                continue

            with np.load(metadata_path) as metadata:
                if not all(key in metadata for key in ("trajectories", "intrinsics", "images")):
                    raise ValueError(f"Missing required keys in {metadata_path}")

                image_names = metadata["images"]
                trajectories = metadata["trajectories"]
                intrinsics = metadata["intrinsics"]
            if not (
                len(image_names) == trajectories.shape[0] == intrinsics.shape[0]
            ):
                raise ValueError(
                    f"Metadata length mismatch in {metadata_path}: "
                    f"images={len(image_names)}, trajectories={trajectories.shape[0]}, "
                    f"intrinsics={intrinsics.shape[0]}"
                )

            frames = []
            for frame_idx, image_name in enumerate(image_names):
                image_name = str(image_name)
                frame_stem = osp.splitext(image_name)[0]
                image_path = osp.join(image_dir, image_name)
                depth_path = osp.join(depth_dir, f"{frame_stem}.png")
                if not osp.isfile(image_path) or not osp.isfile(depth_path):
                    continue

                frames.append(
                    {
                        "metadata_idx": frame_idx,
                        "image_name": image_name,
                        "image_path": image_path,
                        "depth_path": depth_path,
                    }
                )

            if len(frames) < self.min_num_images:
                continue

            scene_infos.append(
                {
                    "scene_name": scene_name,
                    "scene_path": scene_path,
                    "metadata_path": metadata_path,
                    "frames": frames,
                }
            )

        return scene_infos

    def _load_metadata(self, scene_info):
        with np.load(scene_info["metadata_path"]) as metadata:
            return {
                "trajectories": metadata["trajectories"].copy(),
                "intrinsics": metadata["intrinsics"].copy(),
            }

    def _sample_ids(self, num_images: int, img_per_seq: int):
        if img_per_seq is None:
            raise ValueError("img_per_seq must be provided when ids is None")

        if img_per_seq <= 0:
            return np.empty((0,), dtype=np.int64)

        if img_per_seq > num_images and not self.allow_duplicate_img:
            raise ValueError(
                f"Cannot sample {img_per_seq} unique frames from a scene with only {num_images} images"
            )

        return np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

    def _get_extrinsic(self, trajectory: np.ndarray):
        if self.pose_format == "c2w":
            extri_opencv = np.linalg.inv(trajectory)
        else:
            extri_opencv = trajectory
        return extri_opencv[:3].astype(np.float32)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        else:
            seq_index = random.randint(0, self.sequence_list_len - 1) if seq_index is None else seq_index
            seq_index %= self.sequence_list_len

        if seq_name is None:
            scene_info = self.scene_infos[seq_index]
        else:
            scene_info = next(
                (
                    item
                    for item in self.scene_infos
                    if item["scene_path"] == seq_name or item["scene_name"] == seq_name
                ),
                None,
            )
            if scene_info is None:
                raise ValueError(f"Unknown IGGT ScanNet++ scene: {seq_name}")

        frames = scene_info["frames"]
        num_images = len(frames)

        if ids is None:
            ids = self._sample_ids(num_images, img_per_seq)
        else:
            ids = np.asarray(ids, dtype=np.int64)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        metadata = self._load_metadata(scene_info)
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            frame = frames[int(image_idx)]
            metadata_idx = frame["metadata_idx"]
            image_filepath = frame["image_path"]
            depth_filepath = frame["depth_path"]

            image = read_image_cv2(image_filepath)
            depth_map = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
            if depth_map is None:
                raise ValueError(f"Failed to load depth map from {depth_filepath}")

            depth_map = depth_map.astype(np.float32) / self.depth_scale
            depth_map = threshold_depth_map(
                depth_map,
                max_percentile=-1,
                min_percentile=-1,
                max_depth=self.depth_max,
            )

            if image.shape[:2] != depth_map.shape:
                raise ValueError(
                    f"Image and depth shape mismatch for {image_filepath}: "
                    f"{image.shape[:2]} vs {depth_map.shape}"
                )

            original_size = np.array(image.shape[:2])
            intri_opencv = metadata["intrinsics"][metadata_idx].astype(np.float32)
            extri_opencv = self._get_extrinsic(metadata["trajectories"][metadata_idx])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(
                    f"Wrong shape for {scene_info['scene_name']}: "
                    f"expected {target_image_shape}, got {image.shape[:2]}"
                )
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"iggt_scannet_{scene_info['scene_name']}",
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "tracks": None,
            "track_masks": None,
            "sdf": None,
        }
        return batch
