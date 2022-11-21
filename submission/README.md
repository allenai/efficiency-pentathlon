# Simulated Submission to the Efficiency Benchmark
We want to ensure that all submissions use the same data IO and evaluation infrastructures and provide a code skeleton to flesh out. 
This codebase aims to simulate converting the participants' systems into submissions.



## Installation
``` bash
git clone https://github.com/haopeng-nlp/efficiency-benchmark.git
cd efficiency-benchmark/submission
conda env create --file environment.yml
conda activate efficiency-benchmark-submission
```

## To run the MBart machine translation example
``` bash
python inference.py
```

## Simulating the submission process
- `submission.py` lays out the skeleton of a submission. We will ask participants to flesh out this class using their systems and fill in the code blocks marked with `TODO(participants)`. 
- `example_mbart_translation.py` provides an example of an MBart submission using HuggingFace's implementation. 
- Please read the three Python files and try to turn your codebase into a submission (you can change the dataset in `inference.py`). 
- Please record how much effort and time you spend converting your codebase into submission, the challenging/annoying things during the process, and how we can make this process easier. 
- I plan to use the Nov. 21 meeting to collect and discuss the feedback.
