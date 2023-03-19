from typing import Any, Dict, Iterator, Sequence

import more_itertools
import torch
from tango.common import Tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from catwalk.models.template import SubmissionTemplate


class BertExample(SubmissionTemplate):

    def _load_model(self, device):
        ### TODO(participants): load models and necessary tools. ###
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            forced_bos_token_id=0
        ).to(device)

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
                    tensors = self._tokenizer.batch_encode_plus(
                        [instance.text for instance in batch],
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    )
                    tensors = {k: v.to(self._model.device) for k, v in tensors.items()}
                    results = self._model(return_dict=True, **tensors)
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        prediction = self._task.id2label(logits.argmax().item())
                        yield {
                            "label": instance.label,
                            "prediction": self._task.id2label(logits.argmax().item()),
                            "acc": (prediction, instance.label),
                        }
