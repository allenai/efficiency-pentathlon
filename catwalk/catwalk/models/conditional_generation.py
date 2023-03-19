from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import MBartTokenizer, MBartForConditionalGeneration
from catwalk.model import Model
from catwalk.task import InstanceFormat, Task

from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence


@dataclass
class ModelInstance:
    input: str
    target: str


class ConditionalGenerationModel(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.src_lang = "de"
        self.tgt_lang = "en"

    @classmethod
    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(task.instance_conversions[instance_format], instances)

    def prepare(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
    ):
        assert False
        self._task = task
        device = resolve_device()
        self._model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50",
            forced_bos_token_id=0
        ).to(device)
        self._tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")

        self._eval_instances = self._convert_instances(
            [instance["translation"] for instance in instances], 
            InstanceFormat.CONDITIONAL_GENERATION, 
            task
        )

    def predict(
        self,
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        instances = self._eval_instances
        self._model.eval()

        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    input_ids = self._tokenizer(
                        [instance.source for instance in batch], 
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )["input_ids"]
                    input_ids = input_ids.to(self._model.device)
                    # {k: v.to(self._model.device) for k, v in input_ids.items()}

                    outputs = self._model.generate(input_ids)
                    outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    for output in outputs:
                        yield {
                            "output": output
                        }