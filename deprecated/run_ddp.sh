#!/usr/bin/env bash

script=$1
nnode=${2:-1}
npn=2

set -x
torchrun --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nnode $nnode --nproc_per_node $npn --node_rank $NODE_RANK $script