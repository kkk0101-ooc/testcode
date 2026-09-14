#!/usr/bin/env bash
set -euo pipefail
export VITNNS_RUNTIME_ROOT=/data01/home/yeziqi.831/nxtrec_wksp/testcode/.runtime
mkdir -p "$VITNNS_RUNTIME_ROOT"/{tmp,cache,pip,conda_pkgs,hf,hf_datasets,torch,torch_extensions,torchinductor,triton,cuda,matplotlib,numba,pytest,gpu_locks}
export TMPDIR="$VITNNS_RUNTIME_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$VITNNS_RUNTIME_ROOT/cache"
export PIP_CACHE_DIR="$VITNNS_RUNTIME_ROOT/pip"
export CONDA_PKGS_DIRS="$VITNNS_RUNTIME_ROOT/conda_pkgs"
export HF_HOME="$VITNNS_RUNTIME_ROOT/hf"
export HF_DATASETS_CACHE="$VITNNS_RUNTIME_ROOT/hf_datasets"
export TRANSFORMERS_CACHE="$VITNNS_RUNTIME_ROOT/hf/transformers"
export TORCH_HOME="$VITNNS_RUNTIME_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$VITNNS_RUNTIME_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$VITNNS_RUNTIME_ROOT/torchinductor"
export TRITON_CACHE_DIR="$VITNNS_RUNTIME_ROOT/triton"
export CUDA_CACHE_PATH="$VITNNS_RUNTIME_ROOT/cuda"
export MPLCONFIGDIR="$VITNNS_RUNTIME_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$VITNNS_RUNTIME_ROOT/numba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
exec "$@"
