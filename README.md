# efficiency-benchmark


## Installation
``` bash
git clone https://github.com/haopeng-nlp/efficiency-benchmark.git
cd efficiency-benchmark/
bash setup.sh
conda activate efficiency-benchmark
```

## To run the mBART WMT16-enro example
``` bash
python -m catwalk --model mbart --task wmt16-en-ro --split validation
```