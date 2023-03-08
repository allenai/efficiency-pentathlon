from typing import Dict
from catwalk.model import Model
from catwalk.models.bert_example import BertExample
from catwalk.models.submission import Submission
from catwalk.models.gpt import GPTModel
from catwalk.models.conditional_generation import ConditionalGenerationModel


MODELS: Dict[str, Model] = {
    "bert-example": BertExample(),
    "submission": Submission(),
    "gpt2": GPTModel("gpt2"),
    "conditional_generation": ConditionalGenerationModel("mbart")
}
