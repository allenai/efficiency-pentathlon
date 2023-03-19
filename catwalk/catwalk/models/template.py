import warnings
from typing import Any, Dict, Iterator, Sequence, cast, Tuple

import more_itertools
import torch
import time
import numpy as np
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device

from catwalk.model import Model, UnsupportedTaskError
from catwalk.task import InstanceFormat, Task, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import HFClassificationInstance


class SubmissionTemplate(Model):

    def __init__(self):
        self._num_latency_instances = 100

    def _load_model(self, device):
        ### TODO(participants): load models and necessary tools. ###
        # self._tokenizer = XXX
        # self._model = XXX
        raise NotImplementedError()

    def prepare(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
    ) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
        assert isinstance(task, WithAnswerOptionsMixin)
        self._task = task
        device = resolve_device()
        self._load_model(device)
        instances = cast(
            Sequence[HFClassificationInstance],
            self._convert_instances(instances, InstanceFormat.HF_CLASSIFICATION, task))
        indices = list(range(len(instances)))
        np.random.shuffle(indices)
        instances = np.take(instances, indices)
        latency_instances, eval_instances = instances[:self._num_latency_instances], instances[self._num_latency_instances:]
        model_num_labels = self._model.config.num_labels
        if model_num_labels == 1:
            model_num_labels = 2
        if model_num_labels != len(task.answer_options):
            warnings.warn(f"Model has {model_num_labels} labels, but task has {len(task.answer_options)} possible answers.")
        self._model.eval()
        return eval_instances, latency_instances

    def predict(  # type: ignore
        self,
        *,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()

    @classmethod
    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(task.instance_conversions[instance_format], instances)
