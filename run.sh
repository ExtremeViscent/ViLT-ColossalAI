#!/bin/bash

export MASTER_ADDR=$(hostname)
export MASTER_PORT=11455
export NODE_RANK=0

work_dir=$(pwd)
data_root=$work_dir/../../../vilt_data/arrow_coco_mini
cd $work_dir
mpirun -np 4 python run.py with data_root=$data_root num_gpus=1 num_nodes=1 task_mlm_itm_s step200k per_gpu_batchsize=96
# python runcai_opt.py with data_root=/work/zhangyq/vilt_data/arrow_coco_mini num_gpus=1 num_nodes=1 task_mlm_itm_s whole_word_masking=True step200k per_gpu_batchsize=96
