#!/bin/bash

nproc_per_node=${1}
shift

PRO_TMP_DIR=/tmp/pro_metrics/
rm -rf $PRO_TMP_DIR
mkdir -p $PRO_TMP_DIR
export PROMETHEUS_MULTIPROC_DIR=$PRO_TMP_DIR

torchrun --rdzv_backend=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=static --nnodes=${WORLD_SIZE} --nproc_per_node=${nproc_per_node} $@
