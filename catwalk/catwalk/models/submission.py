import warnings
from typing import Any, Dict, Iterator

import more_itertools
import torch
from tango.common import Tqdm

from catwalk.model import UnsupportedTaskError
from catwalk.task import InstanceFormat, Task, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import HFClassificationInstance
from catwalk.models.template import SubmissionTemplate


class Submission(SubmissionTemplate):

    def _load_model(self, device):
        ### TODO(participants): load models and necessary tools. ###
        # self._tokenizer = XXX
        # self._model = XXX
        raise NotImplementedError()

    def predict(  # type: ignore
        self,
        task: Task,
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if not task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            raise UnsupportedTaskError(self, task)
        instances = self._eval_instances
        self._model.eval()

        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    # TODO(participants):  predicts for a minibatch
                    raise NotImplementedError()