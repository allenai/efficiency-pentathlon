import warnings
from typing import Any, Iterator, Dict, List, Sequence, Tuple, cast
import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import (AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer,
                          AutoModelForSequenceClassification,
                          QuestionAnsweringPipeline)
from catwalk.model import Model, UnsupportedTaskError
from catwalk.task import Task, InstanceFormat, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import HFQAInstance, HFMCInstance, HFClassificationInstance


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

    def prepare(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
    ):
        device = resolve_device()
        self._model = self._load_model().to(device)
        self._tokenizer = self._load_tokenizer()
        if task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            self._eval_instances = cast(
                Sequence[HFClassificationInstance],
                self._convert_instances(instances, InstanceFormat.HF_CLASSIFICATION, task))
            assert isinstance(task, WithAnswerOptionsMixin)
            model_num_labels = self._model.config.num_labels
            if model_num_labels == 1:
                model_num_labels = 2
            if model_num_labels != len(task.answer_options):
                warnings.warn(f"Model has {model_num_labels} labels, but task has {len(task.answer_options)} possible answers.")
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            self._eval_instances = cast(Sequence[HFQAInstance], self._convert_instances(instances, InstanceFormat.HF_QA, task))
        elif task.has_instance_conversion(InstanceFormat.HF_MC):
            self._eval_instances = cast(Sequence[HFMCInstance], self._convert_instances(instances, InstanceFormat.HF_MC, task))

    def predict(  # type: ignore
        self,
        task: Task,
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            return self._predict_classification(self._eval_instances, self._model, self._tokenizer, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            return self._predict_qa(self._eval_instances, self._model, self._tokenizer, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_MC):
            return self._predict_mc(self._eval_instances, self._model, self._tokenizer, batch_size=batch_size)
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

    @classmethod
    def _predict_mc(
        cls,
        instances: Sequence[HFMCInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError
        # There is no Huggingface pipeline for this.

        model.eval()
        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    number_of_choices = max(len(instance.answer_choices) for instance in batch)
                    texts: List[Tuple[str, str]] = []
                    labels = []
                    for instance in batch:
                        texts.extend(
                            (instance.question, choice)
                            for choice in instance.answer_choices
                        )
                        while len(texts) % number_of_choices != 0:
                            texts.append(("", ""))  # padding in the choices dimension
                        labels.append(instance.correct_answer_index)
                    tensors = tokenizer.batch_encode_plus(
                        texts,
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    )
                    results = model(
                        return_dict=True,
                        **{
                            key: tensor.view(len(batch), number_of_choices, -1).to(model.device)
                            for key, tensor in tensors.items()
                        })
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        yield {
                            "correct_answer_index": instance.correct_answer_index,
                            "logits": logits,
                            "acc": (logits, instance.correct_answer_index),
                            "relative_improvement": (logits, instance.correct_answer_index),
                        }

    @classmethod
    def _predict_qa(
        cls,
        instances: Sequence[HFQAInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError
        # The type annotation for QuestionAnsweringPipeline says `device` has to be an `int`, but when you look
        # at the code, that's not actually correct.
        pipe = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=model.device)  # type: ignore

        contexts = [instance.context for instance in instances]
        questions = [instance.question for instance in instances]

        pipe_results = pipe(context=contexts, question=questions, batch_size=batch_size)
        with Tqdm.tqdm(pipe_results, desc="Processing instances") as instances_tqdm:
            for instance, prediction in zip(instances, instances_tqdm):
                yield {
                    "squad_metrics": (
                        {"id": instance.id, "prediction_text": prediction["answer"]},
                        {"id": instance.id, "answers": instance.answers}
                    )
                }
