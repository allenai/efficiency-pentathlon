from typing import Any, Dict, Iterator, Optional

import more_itertools
import torch
from tango.common import Tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tango.integrations.torch.util import resolve_device
from catwalk.model import UnsupportedTaskError
from catwalk.models.template import SubmissionTemplate
from catwalk.task import InstanceFormat, Task


class T5(SubmissionTemplate):
    def __init__(self, pretrained_model_name_or_path: str):
        self._pretrained_model_name_or_path = pretrained_model_name_or_path

    def _load_model(self, pretrained_model_name_or_path: str):
        device = resolve_device()
        ### TODO(participants): load models and necessary tools. ###
        self._tokenizer = T5Tokenizer.from_pretrained(self._pretrained_model_name_or_path)
        self._model = T5ForConditionalGeneration.from_pretrained(self._pretrained_model_name_or_path).to(device)


        self._instructions: Optional[Dict[str, str]] = {
            "rte": "You're given a pair of sentences: a Text and a Hypothesis. " \
                + "Your job is to determine the relation between them based on your inference from the statement " \
                + "and your commonsense knowledge. " \
                + f"Answer 'entailment' if the Hypothesis can be inferred from the Text; " \
                + f"Answer 'not_entailment' if the Hypothesis disagrees with the Text.\n",
            "nli": "You're given a pair of sentences: a Premise and a Hypothesis. "\
                + "Your job is to determine the relation between them based on your inference from the statement " \
                + "and your commonsense knowledge. " \
                + f"Answer 'entailment' if the Hypothesis can be inferred from the Premise; " \
                + f"Answer 'neutral' if the Hypothesis disagrees with the Premise; " \
                + f"Answer 'contradiction' if the Hypothesis can neither be inferred from the Premise or disagrees with the Premise.\n",
            "qqp": "You're given a pair of questions. Your job is to determine whether they are duplicate " \
                + f"Answer 'duplicate' if they bear the same meaning; " \
                + f"Answer 'not_duplicate' if they have different meanings.\n"

        }

        if "flan" in self._pretrained_model_name_or_path:
            self._convert_fns = {
                "rte": lambda text: f"{self._instructions['rte']}Text: {text['sentence1']}\nHypothesis: {text['sentence2']}\n",
                "mnli": lambda text: f"{self._instructions['nli']}Premise: {text['premise']}\nHypothesis: {text['hypothesis']}\n",
                "snli": lambda text: f"{self._instructions['nli']}Premise: {text['premise']}\nHypothesis: {text['hypothesis']}\n",
                "qqp": lambda text: f"{self._instructions['qqp']}Question1: {text['question1']}\nQuestion2: {text['question2']}\n"
            }
        else:
            self._convert_fns = {
                "rte": lambda text: f"rte sentence1: {text['sentence1']} sentence2: {text['sentence2']} ",
                "mnli": lambda text: f"mnli hypothesis: {text['hypothesis']} premise: {text['premise']} ",
                "snli": lambda text: f"snli hypothesis: {text['hypothesis']} premise: {text['premise']} ",
                "qqp": lambda text: f"qqp question1: {text['question1']} question2: {text['question2']} "
            }

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

        convert_fn = self._convert_fns[instances[0].task_name]
        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):

                    # TODO(participants):  predicts for a minibatch
                    inputs = [convert_fn(instance.text) for instance in batch]
                    inputs = self._tokenizer.batch_encode_plus(
                        inputs,
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    ).input_ids
                    inputs = inputs.to(self._model.device)
                    outputs = self._model.generate(inputs, max_length=10)
                    outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    for instance, output in zip(batch, outputs):
                        print(instance.label, output)
                        yield {
                            "label": instance.label,
                            "prediction": output.strip(),
                            "acc": (output, instance.label),
                        }
