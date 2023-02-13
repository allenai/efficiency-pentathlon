from typing import Dict
from catwalk.model import Model
from catwalk.models.bert_example import BertExample
from catwalk.models.submission import Submission

MODELS: Dict[str, Model] = {"submission": Submission()}
MODELS: Dict[str, Model] = {"bert-example": BertExample()}
