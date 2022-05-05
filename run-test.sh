#!/bin/bash

nproc_per_node=${1}
shift

PRO_TMP_DIR=/tmp/pro_metrics/
rm -rf $PRO_TMP_DIR
mkdir -p $PRO_TMP_DIR
export PROMETHEUS_MULTIPROC_DIR=$PRO_TMP_DIR

python -m torch.distributed.launch --master_addr="ip-10-0-4-26.ec2.internal" --master_port=${MASTER_PORT} --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=${nproc_per_node} $@
