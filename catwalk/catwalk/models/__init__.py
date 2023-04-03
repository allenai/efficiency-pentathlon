from typing import Dict
from catwalk.model import Model
from catwalk.models.huggingface_classification import HuggingfaceClassification
from catwalk.models.submission import Submission
from catwalk.models.gpt import GPTModel
from catwalk.models.t5 import T5
from catwalk.models.conditional_generation import ConditionalGenerationModel
from catwalk.models.stdio_wrapper import StdioWrapper
from catwalk.models.stdio_docker import StdioDocker


# MODELS: Dict[str, Model] = {
#     # "bert-example": BertExample(),
#     "submission": Submission(),
#     "models/mnli-bert-base": HuggingfaceClassification("models/mnli-bert-base"),
#     "models/mnli-bert-large": HuggingfaceClassification("models/mnli-bert-large"),
#     "models/mnli-deberta-small": HuggingfaceClassification("models/mnli-deberta-small"),
#     "models/mnli-deberta-base": HuggingfaceClassification("models/mnli-deberta-base"),
#     "models/mnli-deberta-large": HuggingfaceClassification("models/mnli-deberta-large"),
#     "models/mnli-roberta-base": HuggingfaceClassification("models/mnli-roberta-base"),
#     "models/mnli-roberta-large": HuggingfaceClassification("models/mnli-roberta-large"),
#     "gpt2": GPTModel("gpt2"),
#     "conditional_generation": ConditionalGenerationModel("mbart"),
#     "t5-small": T5("t5-small"),
#     "models/mnli-t5-small": T5("models/mnli-t5-small"),
#     "t5-base": T5("t5-base"),
#     "models/mnli-t5-base": T5("models/mnli-t5-base"),
#     "t5-large": T5("t5-large"),
#     "models/mnli-t5-large": T5("models/mnli-t5-large"),
#     "t5-3b": T5("t5-3b"),
#     "t5-11b": T5("t5-11b"),
#     "flan-t5-small": T5("google/flan-t5-small"),
#     "flan-t5-base": T5("google/flan-t5-base"),
#     "flan-t5-large": T5("google/flan-t5-large"),
#     "flan-t5-xl": T5("google/flan-t5-xl"),
#     "flan-t5-xxl": T5("google/flan-t5-xxl"),
#     "longformer-base": HuggingfaceClassification("allenai/longformer-base-4096"),
#     "longformer-large": HuggingfaceClassification("allenai/longformer-large-4096"),
#     "distilbert-base": HuggingfaceClassification("distilbert-base-uncased"),
#     "stdio_mbart": StdioWrapper(["/home/haop/miniconda3/envs/efficiency-benchmark/bin/python submission/huggingface/entrypoint.py".split()]),
#     "stdio_docker": StdioDocker(["python3 submission/huggingface/entrypoint.py --model mbart"]),
# }

MODELS: Dict[str, Model] = {
    "mbart": StdioDocker("python3 entrypoint.py --model mbart".split()),
    "flan-t5-small": StdioDocker("python3 entrypoint.py --model google/flan-t5-small".split()),
    "flan-t5-base": StdioDocker("python3 entrypoint.py --model google/flan-t5-base".split()),
    "flan-t5-large": StdioDocker("python3 entrypoint.py --model google/flan-t5-large".split()),
    "flan-t5-xl": StdioDocker("python3 entrypoint.py --model google/flan-t5-xl".split()),
    "flan-t5-xxl": StdioDocker("python3 entrypoint.py --model google/flan-t5-xxl".split()),
    "t5-small": StdioDocker("python3 entrypoint.py --model t5-small --task rte".split()),
    "t5-base": StdioDocker("python3 entrypoint.py --model t5-base".split()),
    "t5-large": StdioDocker("python3 entrypoint.py --model t5-large".split()),
    "t5-3b": StdioDocker("python3 entrypoint.py --model t5-3b".split()),
    "t5-11b": StdioDocker("python3 entrypoint.py --model t5-11b".split()),
}

'["entailment", "entailment", "entailment", "not_entailment", "not_entailment", "not_entailment", "not_entailment", "entailment", "not_entailment", "entailment", "not_entailment", "entailment", "entailment", "entailment", "entailment", "entailment", "not_entailment", "not_entailment", "not_entailment", "entailment", "entailment", "not_entailment", "entailment", "not_entailment", "not_entailment", "not_entailment", "not_entailment", "entailment", "not_entailment", "entailment", "entailment", "not_entailment"]'