#!/bin/bash

PRO_TMP_DIR=/tmp/pro_metrics/
rm -rf $PRO_TMP_DIR
mkdir -p $PRO_TMP_DIR
export PROMETHEUS_MULTIPROC_DIR=$PRO_TMP_DIR

nnode=${WORLD_SIZE:-1}
node_rank=${RANK:-0}

nproc_per_node=${1}
shift
python -m torch.distributed.run --nnode=${nnode} --node_rank=${RANK} --nproc_per_node $num_proc $@
