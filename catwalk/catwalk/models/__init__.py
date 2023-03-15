from typing import Dict
from catwalk.model import Model
from catwalk.models.bert_example import BertExample
from catwalk.models.submission import Submission
from catwalk.models.gpt import GPTModel
from catwalk.models.t5 import T5
from catwalk.models.conditional_generation import ConditionalGenerationModel


MODELS: Dict[str, Model] = {
    "bert-example": BertExample(),
    "submission": Submission(),
    "gpt2": GPTModel("gpt2"),
    "conditional_generation": ConditionalGenerationModel("mbart"),
    "t5-small": T5("t5-small"),
    "t5-base": T5("t5-base"),
    "t5-large": T5("t5-large"),
    "t5-3b": T5("t5-3b"),
    "t5-11b": T5("t5-11b"),
    "flan-t5-small": T5("flan-t5-small"),
    "flan-t5-base": T5("flan-t5-base"),
    "flan-t5-large": T5("flan-t5-large"),
    "flan-t5-xl": T5("flan-t5-xl"),
    "flan-t5-xxl": T5("flan-t5-xxl"),
}
