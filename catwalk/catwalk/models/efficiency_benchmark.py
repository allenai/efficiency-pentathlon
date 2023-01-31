import warnings
from typing import Any, Dict, Iterator, List, Sequence, Tuple, cast

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)

from catwalk import cached_transformers
from catwalk.model import Model, UnsupportedTaskError
from catwalk.task import InstanceFormat, Task, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import (HFClassificationInstance, HFMCInstance)


class BenchmarkModel(Model):
    VERSION = "000"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self._load_model()

    def _load_model(self):
        ### TODO(participants): load models and necessary tools. ###
        return AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            forced_bos_token_id=0
        )
        ### End ###

    def _load_tokenizer(self):
        ### TODO(participants): load tokenizer. ###
        return AutoTokenizer.from_pretrained("bert-base-uncased")
        ### End ###

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        if task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            classification_instances = cast(
                Sequence[HFClassificationInstance],
                self._convert_instances(instances, InstanceFormat.HF_CLASSIFICATION, task))
            model = self._load_model().to(device)
            tokenizer = self._load_tokenizer()
            assert isinstance(task, WithAnswerOptionsMixin)
            model_num_labels = model.config.num_labels
            if model_num_labels == 1:
                model_num_labels = 2
            if model_num_labels != len(task.answer_options):
                warnings.warn(f"Model has {model.config.num_labels} labels, but task has {len(task.answer_options)} possible answers.")
            return self._predict_classification(classification_instances, model, tokenizer, batch_size=batch_size)

        raise UnsupportedTaskError(self, task)

    @classmethod
    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(task.instance_conversions[instance_format], instances)

    @classmethod
    def _predict_classification(
        cls,
        instances: Sequence[HFClassificationInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        # There is no Huggingface pipeline for this.
        # HF's TextClassification pipeline only classifies single texts, not text pairs
        model.eval()
        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    tensors = tokenizer.batch_encode_plus(
                        [instance.text for instance in batch],
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    )
                    tensors = {k: v.to(model.device) for k, v in tensors.items()}
                    results = model(return_dict=True, **tensors)
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        yield {
                            "label": instance.label,
                            "logits": logits,
                            "acc": (logits, instance.label),
                        }
