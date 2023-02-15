# efficiency-benchmark


## Installation
``` bash
git clone https://github.com/haopeng-nlp/efficiency-benchmark.git
cd efficiency-benchmark/
bash setup.sh
conda activate efficiency-benchmark
```

## To run the BERT MNLI example
``` bash
python -m catwalk --model bert-example --task mnli --split validation
```

## Supported tasks:
sst, qqp, qnli, mrpc, mnli, mnli_mismatched, cola, rte, superglue::rte

## Simulating the submission process
- `catwalk/catwalk/models/submission.py` lays out the skeleton of a submission. We will ask participants to flesh out this class using their systems and fill in the code blocks marked with `TODO(participants)`. After implementing it, run
``` bash
python -m catwalk --model submission --task [task] --split [split]
```
- `catwalk/catwalk/models/bert_example.py` provides an example of an BERT submission using HuggingFace's implementation.
- Please record how much effort and time you spend converting your codebase into submission, the challenging/annoying things during the process, and how we can make this process easier. 

## Notes
CPU profiler is temporarily disabled due to a recent change in Beaker access.