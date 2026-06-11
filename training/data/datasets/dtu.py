import logging
import os
import os.path as osp
import random
import re

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import *


class DTU(BaseDataset):
	def __init__(
		self,
		common_conf,
		split: str = "train",
		DTU_DIR: str = "train_data/dtu",
		min_num_images: int = 24,
		len_train: int = 100000,
		len_test: int = 10000,
		expand_ratio: int = 8,
		depth_scale: float = 1000.0,
		depth_max: float = 30.0,
	):
		super().__init__(common_conf=common_conf)

		self.debug = common_conf.debug
		self.training = common_conf.training
		self.get_nearby = common_conf.get_nearby
		self.inside_random = common_conf.inside_random
		self.allow_duplicate_img = common_conf.allow_duplicate_img

		self.expand_ratio = expand_ratio
		self.DTU_DIR = DTU_DIR
		self.min_num_images = min_num_images
		self.depth_scale = depth_scale
		self.depth_max = depth_max
		self.raw_image_size = np.array([1200.0, 1600.0], dtype=np.float32)

		if split == "train":
			self.len_train = len_train
		elif split == "test":
			self.len_train = len_test
		else:
			raise ValueError(f"Invalid split: {split}")

		self.rectified_root = osp.join(self.DTU_DIR, "Rectified")
		self.depth_root = osp.join(self.DTU_DIR, "Depths")
		self.camera_root = osp.join(self.DTU_DIR, "Cameras")

		logging.info(f"DTU_DIR is {self.DTU_DIR}")

		scene_names = sorted(
			scene_name
			for scene_name in os.listdir(self.rectified_root)
			if osp.isdir(osp.join(self.rectified_root, scene_name))
			and osp.isdir(osp.join(self.depth_root, scene_name))
		)
		self.sequence_list = [osp.join(self.rectified_root, scene_name) for scene_name in scene_names]
		self.sequence_list_len = len(self.sequence_list)

		status = "Training" if self.training else "Testing"
		logging.info(f"{status}: DTU Data size: {self.sequence_list_len}")
		logging.info(f"{status}: DTU dataset length: {len(self)}")

	def _read_camera_file(self, camera_filepath: str):
		with open(camera_filepath, "r", encoding="utf-8") as handle:
			lines = [line.strip() for line in handle.readlines() if line.strip()]

		extrinsic = np.array(
			[[float(value) for value in lines[row].split()] for row in range(1, 5)],
			dtype=np.float32,
		)
		intrinsic = np.array(
			[[float(value) for value in lines[row].split()] for row in range(6, 9)],
			dtype=np.float32,
		)
		return extrinsic, intrinsic

	def _read_pfm(self, filepath: str):
		with open(filepath, "rb") as handle:
			header = handle.readline().decode("ascii").rstrip()
			if header not in {"Pf", "PF"}:
				raise ValueError(f"Unsupported PFM header {header} in {filepath}")

			dimension_line = handle.readline().decode("ascii").strip()
			while dimension_line.startswith("#"):
				dimension_line = handle.readline().decode("ascii").strip()

			match = re.match(r"^(\d+)\s+(\d+)$", dimension_line)
			if match is None:
				raise ValueError(f"Malformed PFM dimensions '{dimension_line}' in {filepath}")

			width, height = map(int, match.groups())
			scale = float(handle.readline().decode("ascii").strip())
			dtype = "<f" if scale < 0 else ">f"
			data = np.fromfile(handle, dtype=dtype)

		channel_count = 3 if header == "PF" else 1
		expected_count = width * height * channel_count
		if data.size != expected_count:
			raise ValueError(
				f"PFM size mismatch for {filepath}: expected {expected_count}, got {data.size}"
			)

		shape = (height, width, channel_count) if channel_count == 3 else (height, width)
		data = np.reshape(data, shape)
		data = np.flipud(data)
		return data.astype(np.float32)

	def _scale_intrinsic_to_shape(self, intrinsic: np.ndarray, src_hw, dst_hw):
		src_h, src_w = src_hw
		dst_h, dst_w = dst_hw
		scaled = intrinsic.copy()
		scaled[0, :] *= dst_w / src_w
		scaled[1, :] *= dst_h / src_h
		return scaled

	def _resize_cover_center_crop(self, array: np.ndarray, target_h: int, target_w: int, interpolation: int):
		src_h, src_w = array.shape[:2]
		scale = max(target_w / src_w, target_h / src_h)
		resized_w = int(np.ceil(src_w * scale))
		resized_h = int(np.ceil(src_h * scale))
		resized = cv2.resize(array, (resized_w, resized_h), interpolation=interpolation)

		left = max((resized_w - target_w) // 2, 0)
		top = max((resized_h - target_h) // 2, 0)
		cropped = resized[top : top + target_h, left : left + target_w]
		return cropped, scale, left, top

	def _align_depth_and_intrinsic(self, depth: np.ndarray, intrinsic_for_depth: np.ndarray, rgb_hw):
		target_h, target_w = rgb_hw
		depth_aligned, scale, crop_left, crop_top = self._resize_cover_center_crop(
			depth,
			target_h,
			target_w,
			cv2.INTER_LINEAR,
		)

		intrinsic_aligned = intrinsic_for_depth.copy()
		intrinsic_aligned[0, :] *= scale
		intrinsic_aligned[1, :] *= scale
		intrinsic_aligned[0, 2] -= crop_left
		intrinsic_aligned[1, 2] -= crop_top
		return depth_aligned, intrinsic_aligned

	def _resolve_scene_paths(self, seq_name: str):
		scene_basename = osp.basename(seq_name)
		image_dir = osp.join(self.rectified_root, scene_basename)
		depth_dir = osp.join(self.depth_root, scene_basename)
		return scene_basename, image_dir, depth_dir

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

		if seq_name is None:
			seq_name = self.sequence_list[seq_index]

		scene_basename, image_dir, depth_dir = self._resolve_scene_paths(seq_name)
		num_images = len([name for name in os.listdir(depth_dir) if name.startswith("depth_map_") and name.endswith(".pfm")])

		if num_images < self.min_num_images:
			raise ValueError(f"Scene {scene_basename} has only {num_images} images, smaller than {self.min_num_images}")

		if ids is None:
			ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
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
			lighting_idx = random.randint(0, 6)
			image_id = int(image_idx) + 1
			image_filepath = osp.join(image_dir, f"rect_{image_id:03d}_{lighting_idx}_r5000.png")
			depth_filepath = osp.join(depth_dir, f"depth_map_{int(image_idx):04d}.pfm")
			camera_filepath = osp.join(self.camera_root, f"{int(image_idx):08d}_cam.txt")

			image = read_image_cv2(image_filepath)
			depth_map = self._read_pfm(depth_filepath)
			extri_opencv, intri_raw = self._read_camera_file(camera_filepath)
			intri_depth = self._scale_intrinsic_to_shape(intri_raw, self.raw_image_size, depth_map.shape[:2])
			if image.shape[:2] == depth_map.shape:
				intri_opencv = intri_depth
			else:
				depth_map, intri_opencv = self._align_depth_and_intrinsic(
					depth_map,
					intri_depth,
					image.shape[:2],
				)

			depth_map = depth_map / self.depth_scale
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
			extri_opencv = extri_opencv[:3]
			extri_opencv[:, 3] /= self.depth_scale

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
			"seq_name": f"dtu_{scene_basename}",
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
