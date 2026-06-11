import logging
import os
import os.path as osp
import random

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import *


class TNT(BaseDataset):
	def __init__(
		self,
		common_conf,
		split: str = "train",
		TNT_DIR: str = "train_data/T&T",
		min_num_images: int = 10,
		len_train: int = 100000,
		len_test: int = 10000,
		expand_ratio: int = 8,
		depth_scale: float = 1000.0,
		depth_max: float = 100.0,
	):
		super().__init__(common_conf=common_conf)

		self.debug = common_conf.debug
		self.training = common_conf.training
		self.get_nearby = common_conf.get_nearby
		self.inside_random = common_conf.inside_random
		self.allow_duplicate_img = common_conf.allow_duplicate_img
		self.fixed_scene_images = bool(getattr(common_conf, "fixed_scene_images", False))
		self.expected_scene_image_count = int(getattr(common_conf, "expected_scene_image_count", -1))
		self.fixed_triplet = bool(getattr(common_conf, "fixed_triplet", False))
		self.fixed_triplet_seed = int(getattr(common_conf, "fixed_triplet_seed", 42))
		self.fixed_triplet_mode = str(getattr(common_conf, "fixed_triplet_mode", "first"))
		self.fixed_triplet_ids = getattr(common_conf, "fixed_triplet_ids", None)

		self.expand_ratio = expand_ratio
		self.TNT_DIR = TNT_DIR
		self.min_num_images = min_num_images
		self.depth_scale = depth_scale
		self.depth_max = depth_max

		if split == "train":
			self.len_train = len_train
		elif split == "test":
			self.len_train = len_test
		else:
			raise ValueError(f"Invalid split: {split}")

		logging.info(f"TNT_DIR is {self.TNT_DIR}")

		self.scene_infos = self._build_scene_infos()
		self.sequence_list = [scene_info["scene_path"] for scene_info in self.scene_infos]
		self.sequence_list_len = len(self.sequence_list)

		if self.sequence_list_len == 0:
			raise ValueError(f"No valid T&T scenes found in {self.TNT_DIR}")

		self._validate_scene_image_counts()

		status = "Training" if self.training else "Testing"
		logging.info(f"{status}: T&T Data size: {self.sequence_list_len}")
		logging.info(f"{status}: T&T dataset length: {len(self)}")

	def _build_scene_infos(self):
		scene_infos = []
		for scene_name in sorted(os.listdir(self.TNT_DIR)):
			scene_path = osp.join(self.TNT_DIR, scene_name)
			image_dir = osp.join(scene_path, "images")
			depth_dir = osp.join(scene_path, "depth")
			camera_path = osp.join(scene_path, "sparse", "gt", "cameras.txt")
			images_path = osp.join(scene_path, "sparse", "gt", "images.txt")
			if not (
				osp.isdir(scene_path)
				and osp.isdir(image_dir)
				and osp.isdir(depth_dir)
				and osp.isfile(camera_path)
				and osp.isfile(images_path)
			):
				continue

			camera_dict = self._load_colmap_cameras(camera_path)
			frames = self._load_colmap_frames(images_path)
			valid_frames = []
			for frame in frames:
				frame["intrinsic"] = np.copy(camera_dict[frame["camera_id"]])
				frame["image_path"] = osp.join(image_dir, frame["image_name"])
				frame["depth_path"] = self._resolve_depth_path(depth_dir, frame["image_name"])

				if not osp.isfile(frame["image_path"]):
					logging.warning(f"Skipping missing image file: {frame['image_path']}")
					continue
				if frame["depth_path"] is None:
					logging.warning(
						f"Skipping {scene_name}/{frame['image_name']}: matching depth file not found"
					)
					continue
				valid_frames.append(frame)

			if len(valid_frames) < self.min_num_images:
				logging.warning(
					f"Skipping {scene_name}: only {len(valid_frames)} valid RGB-D frames, "
					f"min_num_images={self.min_num_images}"
				)
				continue

			scene_infos.append(
				{
					"scene_name": scene_name,
					"scene_path": scene_path,
					"frames": valid_frames,
				}
			)

		return scene_infos

	def _validate_scene_image_counts(self):
		if not self.fixed_scene_images:
			return

		if self.expected_scene_image_count <= 0:
			raise ValueError("expected_scene_image_count must be > 0 when fixed_scene_images=True")

		image_counts = [len(scene_info["frames"]) for scene_info in self.scene_infos]
		if any(count != self.expected_scene_image_count for count in image_counts):
			raise ValueError(
				f"T&T fixed-scene-images mode expects exactly {self.expected_scene_image_count} images "
				f"per scene, but found {sorted(set(image_counts))}"
			)

	@staticmethod
	def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
		qw, qx, qy, qz = qvec
		return np.array(
			[
				[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
				[2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
				[2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
			],
			dtype=np.float32,
		)

	def _load_colmap_cameras(self, camera_path: str):
		camera_dict = {}
		with open(camera_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				parts = line.split()
				camera_id = int(parts[0])
				model_name = parts[1]
				params = [float(value) for value in parts[4:]]

				if model_name == "PINHOLE":
					fx, fy, cx, cy = params[:4]
				elif model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
					focal, cx, cy = params[:3]
					fx = focal
					fy = focal
				elif model_name in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
					fx, fy, cx, cy = params[:4]
				else:
					raise ValueError(f"Unsupported COLMAP camera model {model_name} in {camera_path}")

				camera_dict[camera_id] = np.array(
					[
						[fx, 0.0, cx],
						[0.0, fy, cy],
						[0.0, 0.0, 1.0],
					],
					dtype=np.float32,
				)

		if not camera_dict:
			raise ValueError(f"No camera definitions found in {camera_path}")

		return camera_dict

	def _load_colmap_frames(self, images_path: str):
		frames = []
		with open(images_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				parts = line.split()
				if len(parts) < 10:
					continue
				try:
					camera_id = int(parts[8])
				except ValueError:
					continue

				qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float32)
				tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float32)
				image_name = parts[9]
				if osp.splitext(image_name)[1].lower() not in {".jpg", ".jpeg", ".png"}:
					continue
				extri_opencv = np.concatenate(
					[self._qvec_to_rotmat(qvec), tvec.reshape(3, 1)],
					axis=1,
				).astype(np.float32)

				frames.append(
					{
						"camera_id": camera_id,
						"image_name": image_name,
						"extrinsic": extri_opencv,
					}
				)

		frames.sort(key=lambda frame: frame["image_name"])
		return frames

	def _resolve_depth_path(self, depth_dir: str, image_name: str):
		stem = osp.splitext(image_name)[0]
		candidates = []
		try:
			image_id = int(stem)
			candidates.extend(
				[
					osp.join(depth_dir, f"{image_id + 1:06d}.png"),
					osp.join(depth_dir, f"{image_id:06d}.png"),
					osp.join(depth_dir, f"{image_id + 1:08d}.png"),
					osp.join(depth_dir, f"{image_id:08d}.png"),
				]
			)
		except ValueError:
			pass
		candidates.extend(
			[
				osp.join(depth_dir, f"{stem}.png"),
				osp.join(depth_dir, f"{stem}_depth.png"),
			]
		)

		for depth_path in candidates:
			if osp.isfile(depth_path):
				return depth_path
		return None

	def _read_depth(self, depth_path: str):
		depth_map = np.asarray(Image.open(depth_path), dtype=np.float32)
		if depth_map.ndim == 3:
			depth_map = depth_map[..., 0]
		depth_map = depth_map / self.depth_scale
		return threshold_depth_map(
			depth_map,
			max_percentile=-1,
			min_percentile=-1,
			max_depth=self.depth_max,
		)

	def _sample_ids(self, num_images: int, img_per_seq: int):
		if self.fixed_scene_images:
			return np.arange(num_images, dtype=np.int64)
		if img_per_seq is None:
			raise ValueError("img_per_seq must be provided when fixed_scene_images=False")
		if img_per_seq <= 0:
			return np.empty((0,), dtype=np.int64)
		if img_per_seq > num_images and not self.allow_duplicate_img:
			raise ValueError(
				f"Cannot sample {img_per_seq} unique frames from a scene with only {num_images} images"
			)
		if img_per_seq == num_images and not self.allow_duplicate_img:
			return np.random.permutation(num_images).astype(np.int64)
		return np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img).astype(np.int64)

	def _fixed_ids(self, seq_index: int, num_images: int, img_per_seq: int):
		if img_per_seq is None:
			raise ValueError("img_per_seq must be provided when fixed_triplet=True")
		if img_per_seq <= 0:
			return np.empty((0,), dtype=np.int64)
		if self.fixed_triplet_ids is not None:
			ids = np.asarray(self.fixed_triplet_ids, dtype=np.int64)
			if len(ids) != img_per_seq:
				raise ValueError(f"fixed_triplet_ids length must equal img_per_seq={img_per_seq}, got {len(ids)}")
			if (ids < 0).any() or (ids >= num_images).any():
				raise ValueError(f"fixed_triplet_ids out of range for scene with {num_images} images: {ids.tolist()}")
			return ids
		if img_per_seq > num_images and not self.allow_duplicate_img:
			raise ValueError(
				f"Cannot choose fixed {img_per_seq} unique frames from a scene with only {num_images} images"
			)
		if self.fixed_triplet_mode == "first":
			if img_per_seq <= num_images:
				return np.arange(img_per_seq, dtype=np.int64)
			return np.arange(img_per_seq, dtype=np.int64) % num_images
		if self.fixed_triplet_mode == "uniform":
			if img_per_seq == 1:
				return np.array([0], dtype=np.int64)
			return np.linspace(0, num_images - 1, img_per_seq).round().astype(np.int64)
		if self.fixed_triplet_mode == "seeded":
			rng = np.random.default_rng(self.fixed_triplet_seed + int(seq_index))
			return rng.choice(num_images, img_per_seq, replace=self.allow_duplicate_img).astype(np.int64)
		raise ValueError(
			f"Unsupported fixed_triplet_mode={self.fixed_triplet_mode!r}; use first, uniform, or seeded"
		)

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
			seq_name_to_info = {scene_info["scene_path"]: scene_info for scene_info in self.scene_infos}
			scene_info = seq_name_to_info[seq_name]

		scene_basename = scene_info["scene_name"]
		frames = scene_info["frames"]
		num_images = len(frames)

		if ids is None and self.fixed_triplet and not self.fixed_scene_images:
			ids = self._fixed_ids(seq_index, num_images, img_per_seq)
		elif ids is None:
			ids = self._sample_ids(num_images, img_per_seq)
		else:
			ids = np.asarray(ids, dtype=np.int64)

		if self.fixed_scene_images:
			if len(ids) != num_images or len(np.unique(ids)) != num_images:
				raise ValueError(
					f"T&T fixed-scene-images mode requires all {num_images} unique images from scene {scene_basename}"
				)
			ids = np.sort(ids)
		elif self.get_nearby and not self.fixed_triplet:
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
			frame = frames[int(image_idx)]
			image_filepath = frame["image_path"]
			depth_filepath = frame["depth_path"]

			image = read_image_cv2(image_filepath)
			if image is None:
				raise ValueError(f"Failed to load image from {image_filepath}")
			depth_map = self._read_depth(depth_filepath)

			if image.shape[:2] != depth_map.shape:
				raise ValueError(
					f"Image and depth shape mismatch for {image_filepath}: {image.shape[:2]} vs {depth_map.shape}"
				)

			original_size = np.array(image.shape[:2])
			intri_opencv = np.copy(frame["intrinsic"])
			extri_opencv = np.copy(frame["extrinsic"])

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
			depths.append(depth_map.astype(np.float32))
			extrinsics.append(extri_opencv)
			intrinsics.append(intri_opencv)
			cam_points.append(cam_coords_points.astype(np.float32))
			world_points.append(world_coords_points.astype(np.float32))
			point_masks.append(point_mask)
			original_sizes.append(original_size)

		batch = {
			"seq_name": f"tnt_{scene_basename}",
			"ids": ids,
			"frame_num": len(extrinsics),
			"images": images,
			"depths": depths,
			"extrinsics": extrinsics,
			"intrinsics": intrinsics,
			"cam_points": cam_points,
			"world_points": world_points,
			"point_masks": point_masks,
			"camera_valid_mask": np.array(True, dtype=bool),
			"has_depth": np.array(True, dtype=bool),
			"original_sizes": original_sizes,
			"tracks": None,
			"track_masks": None,
			"sdf": None,
		}
		return batch


TandT = TNT
