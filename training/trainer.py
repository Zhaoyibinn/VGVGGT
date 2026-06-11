# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import inspect
import json
import logging
import math
import time
from datetime import timedelta
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers

from tqdm import tqdm




class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()
        self._setup_tb_state()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim
        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        resumed_from_checkpoint = False
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
            resumed_from_checkpoint = True
        elif getattr(self.checkpoint_conf, "auto_resume", True):
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)
                resumed_from_checkpoint = True

        self.train_tb_writer = self._build_tb_writer(
            purge_step=self._train_tb_purge_step() if resumed_from_checkpoint else None,
            log_all_ranks=True,
        )
        self.val_tb_writer = self._build_tb_writer(
            purge_step=self.steps.get("val") if resumed_from_checkpoint else None,
        )
        self.tb_writer = self.train_tb_writer

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_tb_state(self):
        """Initializes TensorBoard step bookkeeping."""
        self.train_tb_resume_step = 0
        self.train_tb_prev_world_size = 1

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins),
            device_id=self.device if self.device.type == "cuda" else None,
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            # logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
            logging.info(f"Model state loaded. Missing keys: {len(missing) or 'None'}. Unexpected keys: {len(unexpected) or 'None'}.")
        # Load optimizer state if available and in training mode
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None and self.mode != "val":
            optimizer_states = optimizer_state if isinstance(optimizer_state, list) else [optimizer_state]
            # if len(optimizer_states) != len(self.optims):
            #     logging.warning(
            #         "Skipping optimizer restore because checkpoint has %d optimizer states but trainer has %d optimizers.",
            #         len(optimizer_states),
            #         len(self.optims),
            #     )
            # else:
            for optim, state_dict in zip(self.optims, optimizer_states):
                optim.optimizer.load_state_dict(state_dict)
            logging.info("Optimizer state restored from checkpoint.")
        else:
            logging.info("No optimizer state restored from checkpoint.")

        # Load training progress
        if "prev_epoch" in checkpoint:
            self.epoch = int(checkpoint["prev_epoch"]) + 1
        elif "epoch" in checkpoint:
            self.epoch = int(checkpoint["epoch"])
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
        tb_state = checkpoint.get("tb_state", {})
        self.train_tb_resume_step = int(
            tb_state.get("train_resume_step", self.steps.get("train", 0))
        )
        self.train_tb_prev_world_size = int(tb_state.get("train_world_size", 1))

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        logging.info(
            "Resumed training progress: epoch=%s, train_step=%s, val_step=%s, tb_resume_step=%s, tb_prev_world_size=%s",
            self.epoch,
            self.steps.get("train", 0),
            self.steps.get("val", 0),
            self.train_tb_resume_step,
            self.train_tb_prev_world_size,
        )

    def _build_tb_writer(
        self,
        purge_step: Optional[int] = None,
        *,
        log_all_ranks: bool = False,
        rank_subdir: bool = False,
        path_suffix: Optional[str] = None,
    ):
        filename_suffix = getattr(self.logging_conf, "tensorboard_filename_suffix", None)
        if not filename_suffix:
            filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        writer_kwargs = {}
        if purge_step is not None:
            writer_kwargs["purge_step"] = int(purge_step)

        writer_path = self.logging_conf.tensorboard_writer.path
        if path_suffix:
            writer_path = os.path.join(writer_path, path_suffix)

        return instantiate(
            self.logging_conf.tensorboard_writer,
            path=writer_path,
            filename_suffix=filename_suffix,
            log_all_ranks=log_all_ranks,
            rank_subdir=rank_subdir,
            _recursive_=False,
            **writer_kwargs,
        )

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = None
        self.model = instantiate(self.model_conf, _recursive_=False)
        # max_depth = 4
        # for name, mod in self.model.named_modules():
        #     if name and (name.count(".") + 1) <= max_depth:
        #         print(name)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        amp_dtype = getattr(self.optim_conf.amp, "amp_dtype", "float16")
        self.use_grad_scaler = self.optim_conf.amp.enabled and amp_dtype == "float16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        self._log_parameter_status()

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            # model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _log_parameter_status(self):
        """Logs total, frozen, and trainable parameter counts for the current model."""
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        frozen_params = total_params - trainable_params

        if total_params == 0:
            logging.info("Parameter status: model has no parameters.")
            return

        trainable_ratio = 100.0 * trainable_params / total_params
        frozen_ratio = 100.0 * frozen_params / total_params
        total_params_b = total_params / 1e9
        trainable_params_b = trainable_params / 1e9
        frozen_params_b = frozen_params / 1e9

        logging.info(
            "Parameter status on rank %s: total=%.6f B, trainable=%.6f B (%.2f%%), frozen=%.6f B (%.2f%%)",
            self.distributed_rank,
            total_params_b,
            trainable_params_b,
            trainable_ratio,
            frozen_params_b,
            frozen_ratio,
        )

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )
        init_sync = getattr(distributed_conf, "init_sync", True)
        ddp_parameters = inspect.signature(
            nn.parallel.DistributedDataParallel.__init__
        ).parameters
        if "init_sync" in ddp_parameters:
            ddp_options["init_sync"] = init_sync
        elif not init_sync:
            logging.warning(
                "This PyTorch version does not support DDP init_sync=False; "
                "parameters will be synchronized during DDP initialization."
            )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "tb_state": {
                "train_resume_step": int(self.steps.get("train", 0)),
                "train_world_size": int(self._get_world_size()),
            },
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_world_size(self) -> int:
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1

    def _get_global_progress_total(self, data_loader, limit_batches: Optional[int]) -> int:
        local_total = len(data_loader)
        if limit_batches is not None:
            local_total = min(int(limit_batches), local_total)
        return int(local_total * self._get_world_size())

    def _get_global_batch_size(self, batch: Mapping) -> int:
        local_batch_size = int(batch["images"].shape[0])
        if self._get_world_size() == 1:
            return local_batch_size

        batch_size_tensor = torch.tensor(
            local_batch_size, device=self.device, dtype=torch.long
        )
        dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.SUM)
        return int(batch_size_tensor.item())

    def _distributed_gather_scalars(self, value: float, count: int = 1) -> List[tuple[int, float, int]]:
        if self._get_world_size() == 1:
            return [(0, float(value), int(count))]

        payload = torch.tensor(
            [float(value), float(count)],
            device=self.device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(payload) for _ in range(self._get_world_size())]
        dist.all_gather(gathered, payload)
        return [
            (rank, float(item[0].item()), int(item[1].item()))
            for rank, item in enumerate(gathered)
        ]

    def _tb_scalar_name(self, phase: str, key: str, *, epoch_average: bool = False) -> str:
        return f"Values/{phase}/{key}"

    def _train_tb_purge_step(self) -> int:
        return int(self.train_tb_resume_step * self.train_tb_prev_world_size)

    def _train_tb_step(self, step: int) -> int:
        step = int(step)
        relative_step = max(0, step - self.train_tb_resume_step)
        base_step = self._train_tb_purge_step()
        current_world_size = self._get_world_size()

        if current_world_size == 1:
            return base_step + relative_step
        return base_step + relative_step * current_world_size + self.rank

    def _distributed_average_scalar(self, value: float, count: int = 1) -> float:
        if count <= 0:
            return 0.0
        if self._get_world_size() == 1:
            return float(value)

        payload = torch.tensor(
            [float(value) * float(count), float(count)],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        total_count = float(payload[1].item())
        if total_count <= 0:
            return 0.0
        return float(payload[0].item() / total_count)

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            # self.run_val()
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        pbar = tqdm(
            total=self._get_global_progress_total(val_loader, limit_val_batches),
            desc="Validating",
            disable=self.rank != 0,
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            pbar.update(self._get_global_batch_size(batch))
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters, log_scalars=False
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # if data_iter % self.logging_conf.log_freq == 0:
            #     progress.display(data_iter)


        if loss_meters:
            self._log_epoch_scalars_to_tb(phase, loss_meters, self.steps[phase])
        pbar.close()
        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        acc_loss_R = 0.0
        acc_loss_depth = 0.0

        pbar = tqdm(
            total=self._get_global_progress_total(train_loader, limit_train_batches),
            desc="Training",
            disable=self.rank != 0,
        )

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            pbar.update(self._get_global_batch_size(batch))
            self.data_iter = data_iter
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            if True:
                images = batch['images']
                depths = batch['depths']
                exts = batch['extrinsics']
                ints = batch['intrinsics']

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            did_backward = self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            if not did_backward:
                for optim in self.optims:
                    optim.zero_grad(set_to_none=True)

                batch_time.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )
                mem.update(torch.cuda.max_memory_allocated() // 1e9)
                continue

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.rank == 0 and self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    if self.use_grad_scaler:
                        self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:
                if self.use_grad_scaler:
                    self.scaler.step(optim.optimizer)
                else:
                    optim.optimizer.step()
            if self.use_grad_scaler:
                self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)
            acc_loss_R += loss_meters['Loss/train_loss_R'].value
            acc_loss_depth += loss_meters['Loss/train_loss_reg_depth'].value
            # if data_iter % self.logging_conf.log_freq == 0:
            #     print(f"Train Epoch:[{self.epoch}]; Iteration:{data_iter}; Depth:{loss_meters['Loss/train_loss_reg_depth'].value}; R:{loss_meters['Loss/train_loss_R'].value}")
            #     progress.display(data_iter)
        if self.rank == 0:
            print(f"Epoch {self.epoch} Average Depth:{acc_loss_depth/(data_iter+1)}; R:{acc_loss_R/(data_iter+1)}")
        pbar.close()
        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ) -> bool:
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = (
                        f"Loss is {loss.item()}, skipping optimizer step for this batch"
                    )
                    logging.error(error_msg)
                    return False

                loss /= accum_steps
                if self.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

        return True


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)

        for key in ("camera_valid_mask", "has_depth"):
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, original_tensor], dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, normalized_voxel_xyz = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
                voxel_xyz=batch.get("voxel_xyz", None),
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths
        if "voxel_xyz" in batch:
            batch["voxel_xyz"] = normalized_voxel_xyz

        return batch

    def _step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict,
        log_scalars: bool = True,
    ):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        if model.training:
            y_hat = model(images=batch["images"],gt_data = batch)
        else:
            y_hat = model(images=batch["images"])
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch,train = model.training)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(
            log_data, phase, self.steps[phase], loss_meters, log_to_tb=log_scalars
        )
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(
        self,
        data: Mapping,
        phase: str,
        step: int,
        loss_meters: dict,
        log_to_tb: bool = True,
    ):
        """Updates average meters and optionally logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if log_to_tb and step % self.logging_conf.log_freq == 0:
                    self.train_tb_writer.log(
                        self._tb_scalar_name(phase, key),
                        value,
                        self._train_tb_step(step),
                    )

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if self.rank != 0:
            return
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )

    def _log_epoch_scalars_to_tb(
        self, phase: str, loss_meters: Dict[str, AverageMeter], step: int
    ) -> None:
        """Logs globally averaged scalars for the phase to TensorBoard once per epoch."""
        if not self.logging_conf.scalar_keys_to_log:
            return

        keys_to_log = self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        for key in keys_to_log:
            meter = loss_meters.get(f"Loss/{phase}_{key}")
            value = meter.avg if meter is not None and meter.count > 0 else 0.0
            count = meter.count if meter is not None else 0
            global_value = self._distributed_average_scalar(value, count)
            if self.rank != 0:
                continue
            self.val_tb_writer.log(
                self._tb_scalar_name(phase, key, epoch_average=True),
                global_value,
                step,
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
