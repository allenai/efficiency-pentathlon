from typing import Any, Dict, Iterator, Sequence

import more_itertools
import torch
from tango.common import Tqdm

from catwalk.models.template import SubmissionTemplate


class Submission(SubmissionTemplate):

    def load_model(self):
        ### TODO(participants): load models and necessary tools. ###
        # self.tokenizer = XXX
        # self.model = XXX
        raise NotImplementedError()

    def predict(  # type: ignore
        self,
        *,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:

        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    # TODO(participants):  predicts for a minibatch
                    raise NotImplementedError()