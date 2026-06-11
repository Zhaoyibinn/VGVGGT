import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import *


class Replica(BaseDataset):
	def __init__(
		self,
		common_conf,
		split: str = "train",
		REPLICA_DIR: str = "train_data/replica",
		min_num_images: int = 24,
		len_train: int = 100000,
		len_test: int = 10000,
		expand_ratio: int = 8,
		min_frame_interval: int = 10,
		depth_scale: float = 6553.5,
		depth_max: float = 30.0,
		fx: float = 600.0,
		fy: float = 600.0,
		cx: float = 599.5,
		cy: float = 339.5,
	):
		super().__init__(common_conf=common_conf)

		self.debug = common_conf.debug
		self.training = common_conf.training
		self.get_nearby = common_conf.get_nearby
		self.inside_random = common_conf.inside_random
		self.allow_duplicate_img = common_conf.allow_duplicate_img

		self.expand_ratio = expand_ratio
		self.min_frame_interval = min_frame_interval
		self.REPLICA_DIR = REPLICA_DIR
		self.min_num_images = min_num_images
		self.depth_scale = depth_scale
		self.depth_max = depth_max
		self.default_intrinsic = np.array(
			[
				[fx, 0.0, cx],
				[0.0, fy, cy],
				[0.0, 0.0, 1.0],
			],
			dtype=np.float32,
		)

		if split == "train":
			self.len_train = len_train
		elif split == "test":
			self.len_train = len_test
		else:
			raise ValueError(f"Invalid split: {split}")

		logging.info(f"REPLICA_DIR is {self.REPLICA_DIR}")

		scene_names = sorted(
			scene_name
			for scene_name in os.listdir(self.REPLICA_DIR)
			if osp.isdir(osp.join(self.REPLICA_DIR, scene_name))
			and osp.isdir(osp.join(self.REPLICA_DIR, scene_name, "results"))
			and osp.isfile(osp.join(self.REPLICA_DIR, scene_name, "traj.txt"))
		)
		self.sequence_list = [osp.join(self.REPLICA_DIR, scene_name) for scene_name in scene_names]
		self.sequence_list_len = len(self.sequence_list)

		status = "Training" if self.training else "Testing"
		logging.info(f"{status}: Replica Data size: {self.sequence_list_len}")
		logging.info(f"{status}: Replica dataset length: {len(self)}")

	def _load_trajectory(self, traj_filepath: str):
		poses = np.loadtxt(traj_filepath, dtype=np.float32)
		poses = poses.reshape(-1, 4, 4)
		return poses

	def _resolve_scene(self, seq_name: str):
		scene_basename = osp.basename(seq_name)
		result_dir = osp.join(seq_name, "results")
		traj_filepath = osp.join(seq_name, "traj.txt")
		return scene_basename, result_dir, traj_filepath

	def _sample_ids_with_min_interval(self, num_images: int, img_per_seq: int) -> np.ndarray:
		if img_per_seq is None:
			raise ValueError("img_per_seq must be provided when ids is None")

		if img_per_seq <= 0:
			return np.empty((0,), dtype=np.int64)

		if self.min_frame_interval <= 0:
			return np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

		max_selectable = (num_images + self.min_frame_interval - 1) // self.min_frame_interval
		if img_per_seq > max_selectable:
			raise ValueError(
				f"Cannot sample {img_per_seq} frames from {num_images} images with minimum interval {self.min_frame_interval}"
			)

		candidate_ids = np.random.permutation(num_images)
		selected_ids = []
		for candidate_id in candidate_ids:
			if all(abs(int(candidate_id) - existing_id) >= self.min_frame_interval for existing_id in selected_ids):
				selected_ids.append(int(candidate_id))
				if len(selected_ids) == img_per_seq:
					break

		if len(selected_ids) != img_per_seq:
			raise ValueError(
				f"Failed to sample {img_per_seq} frames with minimum interval {self.min_frame_interval} from {num_images} images"
			)

		return np.asarray(selected_ids, dtype=np.int64)

	def get_data(
		self,
		seq_index: int = None,
		img_per_seq: int = None,
		seq_name: str = None,
		ids: list = None,
		aspect_ratio: float = 1.0,
	) -> dict:
		if self.sequence_list_len == 0:
			raise ValueError(f"No valid Replica scenes found in {self.REPLICA_DIR}")

		if self.inside_random and self.training:
			seq_index = random.randint(0, self.sequence_list_len - 1)
		else:
			seq_index = random.randint(0, self.sequence_list_len - 1) if seq_index is None else seq_index
			seq_index %= self.sequence_list_len

		if seq_name is None:
			seq_name = self.sequence_list[seq_index]

		scene_basename, result_dir, traj_filepath = self._resolve_scene(seq_name)
		traj = self._load_trajectory(traj_filepath)
		num_images = traj.shape[0]

		if num_images < self.min_num_images:
			raise ValueError(f"Scene {scene_basename} has only {num_images} images, smaller than {self.min_num_images}")

		if ids is None:
			ids = self._sample_ids_with_min_interval(num_images, img_per_seq)
		else:
			ids = np.asarray(ids)

		if self.get_nearby:
			ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

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
			frame_idx = int(image_idx)
			image_filepath = osp.join(result_dir, f"frame{frame_idx:06d}.jpg")
			depth_filepath = osp.join(result_dir, f"depth{frame_idx:06d}.png")

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
					f"Image and depth shape mismatch for {image_filepath}: {image.shape[:2]} vs {depth_map.shape}"
				)

			original_size = np.array(image.shape[:2])
			intri_opencv = np.copy(self.default_intrinsic)
			extri_opencv = np.linalg.inv(traj[frame_idx])[:3]

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
					f"Wrong shape for {scene_basename}: expected {target_image_shape}, got {image.shape[:2]}"
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
			"seq_name": f"replica_{scene_basename}",
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