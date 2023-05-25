# efficiency-benchmark


## Installation
``` bash
git clone https://github.com/haopeng-nlp/efficiency-benchmark.git
cd efficiency-benchmark/
pip install .
```

## To run the mBART WMT16-enro example locally
``` 
https://github.com/haopeng-nlp/submission.git
cd submission
pip install -r requirements.txt
efficiency-benchmark run --task wmt16-en-ro  --max_batch_size 100 --scenario accuracy  -- python entrypoint.py --model mbart
```

## To submit to the dedicated machine
```
efficiency-benchmark submit --task wmt16-en-ro  --max_batch_size 100  -- python entrypoint.py --model mbart
```

Please contact haop@ to get access to this machine.