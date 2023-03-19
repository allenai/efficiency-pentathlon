from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from catwalk.model import Model
from catwalk.task import InstanceFormat, Task

from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence


@dataclass
class ModelInstance:
    text: str
    num_context_tokens: int
    input_ids: torch.Tensor
    targets: torch.Tensor


class GPTModel(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(lambda instance: task.convert_instance(instance, instance_format), instances)

    def prepare(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
    ):
        assert False
        self._tasl = task
        # TODO: max_length
        device = resolve_device()
        self._model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).eval().to(device)
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # TODO: This should be specified by us, such that all models use the same context size and are comparable.
        self._max_length = self._tokenizer.model_max_length
        def make_model_instances(
            texts: Iterator[str],
            overlap: int = 1
        ) -> Iterator[ModelInstance]:
            for text in texts:
                token_ids = [self._tokenizer.eos_token_id] + self._tokenizer.encode(text)
                # The next line puts the entire text into GPU memory. In principle this is a problem, because it
                # might OOM when the text is long. In practice, that doesn't happen.
                token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
                window_start = 0
                while True:
                    window_end = window_start + self._max_length
                    if window_end > len(token_ids) - 1:
                        break
                    yield ModelInstance(
                        text,
                        1 if window_start == 0 else overlap,
                        token_ids[window_start:window_end],
                        token_ids[window_start+1:window_end+1])
                    window_start += self._max_length
                    window_start -= overlap
                window_end = len(token_ids) - 1
                if window_start == 0:
                    last_batch_context_tokens = 1
                else:
                    new_window_start = window_end - self._max_length
                    last_batch_context_tokens = window_start - new_window_start + overlap
                    window_start = new_window_start
                    del new_window_start
                yield ModelInstance(
                    text,
                    last_batch_context_tokens,
                    token_ids[window_start:window_end],
                    token_ids[window_start+1:window_end+1])
        
        # def make_model_instances(
        #     texts: Iterator[str],
        #     overlap: int = 1
        # ) -> Iterator[ModelInstance]:

        #     # TODO: this is wikitext specific
        #     texts = "\n\n".join(texts).split()
        #     seq_len = len(texts)
        #     stride = min(1, self._max_length - overlap)
        #     for window_start in Tqdm.tqdm(range(0, seq_len, stride)):
        #         window_end = min(window_start + self._max_length, seq_len)
        #         print(texts[window_start: window_end])
        #         yield " ".join(texts[window_start: window_end])

        self._eval_instances = make_model_instances(
            task.convert_instance(instance, InstanceFormat.ELEUTHER_REQUESTS).args[0] for instance in Tqdm.tqdm(
                instances,
                desc="Calculating log probabilities")
        )
        # self._eval_instances = make_model_instances(
        #     instances,
        #     overlap = 256
        # )


    def group_model_predictions(
        self,
        model_predictions: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, float]]:
        last_text = None
        summed_logprobs = 0.0
        for text, logprobs in model_predictions:
            if last_text is not None and text != last_text:
                yield last_text, float(summed_logprobs)
                summed_logprobs = 0.0
            summed_logprobs += logprobs.sum()
            last_text = text
        if last_text is not None:
            yield last_text, float(summed_logprobs)

    def predict(
        self,
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:

        def make_model_predictions(model_instances: Iterator[ModelInstance]) -> Iterator[Tuple[str, torch.Tensor]]:
            for batch in more_itertools.chunked(model_instances, batch_size):
                batch_results = []
                with torch.inference_mode():
                    inputs = pad_sequence(
                        [mi.input_ids for mi in batch], batch_first=True)
                    # print(inputs)
                    outputs = self._model(inputs)
                    outputs = log_softmax(outputs.logits, dim=-1)
                    for mi, output in zip(batch, outputs):
                        # gets rid of padding
                        output = output[:len(mi.targets)]
                        logprobs = torch.gather(
                            output[mi.num_context_tokens:],
                            1,
                            mi.targets[mi.num_context_tokens:].unsqueeze(-1)).squeeze(-1)
                        batch_results.append((mi.text, logprobs))
                yield from batch_results

        model_predictions = make_model_predictions(self._eval_instances)
        grouped_predictions = self.group_model_predictions(model_predictions)

        from spacy.lang.en import English
        spacy_tokenizer = English().tokenizer
        for text, logprob in grouped_predictions:
            yield {
                "text": text,
                "word_perplexity": (logprob, len(spacy_tokenizer(text))),
                # bytes aren't characters, but this is what Eleuther calls it
                "byte_perplexity": (logprob, len(text)),
                "bits_per_byte": (logprob, len(text))
            }

