#! /bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate efficiency-benchmark

# supported tasks:
# sst
# qqp
# qnli
# mrpc
# mnli
# mnli_mismatched
# cola
# rte
# superglue::rte

python -m catwalk --model bert-example --task mnli --split validation