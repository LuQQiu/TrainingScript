#!/bin/bash

nproc_per_node=${1}
shift

PRO_TMP_DIR=/tmp/pro_metrics/
rm -rf $PRO_TMP_DIR
mkdir -p $PRO_TMP_DIR
export PROMETHEUS_MULTIPROC_DIR=$PRO_TMP_DIR

if [[ ${MASTER_ADDR} != "localhost" ]]; then
   sleep(100000)
fi
python -m torch.distributed.launch --master_addr="${MASTER_ADDR}" --master_port=${MASTER_PORT} --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=${nproc_per_node} $@
