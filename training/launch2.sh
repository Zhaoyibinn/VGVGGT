export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cd "${REPO_ROOT}"
# export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=2 training/launch.py --config demo
