#!/usr/bin/env bash
torchrun --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nnode 2 --nproc_per_node 2 --node_rank $NODE_RANK train_moco.py