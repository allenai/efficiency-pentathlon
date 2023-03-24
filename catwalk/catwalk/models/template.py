from typing import Any, Dict, Iterator, Sequence, cast, Tuple

import numpy as np
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device

from catwalk.model import Model
from catwalk.task import InstanceFormat, Task, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import HFClassificationInstance


class SubmissionTemplate(Model):

    def __init__(self):
        self.num_latency_instances = 100

        # TODO(haop)
        self.instance_format = InstanceFormat.HF_CLASSIFICATION

    def load_model(self):
        ### TODO(participants): load models and necessary tools. ###
        # self.tokenizer = XXX
        # self.model = XXX
        raise NotImplementedError()

    # def prepare(
    #     self,
    #     task: Task,
    #     instances: Sequence[Dict[str, Any]],
    # ) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    #     assert isinstance(task, WithAnswerOptionsMixin)
    #     self.task = task
    #     self.load_model()
    #     eval_instances = self._convert_instances(
    #         instances, InstanceFormat.HF_CLASSIFICATION, task)
    #     indices = list(range(len(instances)))
    #     np.random.shuffle(indices)
    #     indices = indices[:self.num_latency_instances]
    #     latency_instances = np.take(eval_instances, indices)
    #     self.model.eval()
    #     return eval_instances, latency_instances

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
