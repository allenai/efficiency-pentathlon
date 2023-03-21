from typing import Any, Dict, Iterator, Sequence

import more_itertools
import torch
from tango.common import Tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tango.integrations.torch.util import resolve_device

from catwalk.models.template import SubmissionTemplate


class HuggingfaceClassification(SubmissionTemplate):

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        SubmissionTemplate.__init__(self)

    def load_model(self):
        device = resolve_device()
        ### TODO(participants): load models and necessary tools. ###
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name_or_path,
            forced_bos_token_id=0
        ).to(device)
        self._convert_fn = lambda text: " ".join(text[k] for k in text.keys())

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
                    tensors = self.tokenizer.batch_encode_plus(
                        [self._convert_fn(instance.text) for instance in batch],
                        padding=True,
                        # truncation="only_first",
                        return_tensors="pt",
                        #pad_to_multiple_of=8,
                        truncation=False,
                    )
                    tensors = {k: v.to(self.model.device) for k, v in tensors.items()}
                    results = self.model(return_dict=True, **tensors)
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        prediction = self.task.id2label(logits.argmax().item())
                        yield {
                            "label": instance.label,
                            "prediction": prediction,
                            "acc": (prediction, instance.label),
                        }
