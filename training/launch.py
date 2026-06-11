# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path

# Allow direct execution via Python, torchrun, or VS Code debugpy.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import shutil
import os


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    exp_dir = cfg.logging.log_dir +  "/" + cfg.exp_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    print("Experiment Directory:", exp_dir)
    config_root = "training/config"
    yaml_copy_path = exp_dir +  "/" + f"{args.config}.yaml"
    yaml_path = config_root + "/" + f"{args.config}.yaml"
    shutil.copyfile(yaml_path, yaml_copy_path)
    print(f"Copied config file to {yaml_copy_path}")
    default_yaml_copy_path = exp_dir +  "/" + "default_dataset.yaml" 
    default_yaml_path = config_root + "/" + "default_dataset.yaml" 
    shutil.copyfile(default_yaml_path, default_yaml_copy_path)
    print(f"Copied default config file to {default_yaml_copy_path}")


    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()

