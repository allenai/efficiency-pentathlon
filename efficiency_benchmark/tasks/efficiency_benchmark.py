import functools
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import datasets
from tango.common.sequences import MappedSequence

from efficiency_benchmark.task import InstanceConversion, Task
from efficiency_benchmark.tasks.huggingface import get_from_dict


class EfficiencyBenchmarkTask(Task):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        Task.__init__(self, version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    def dataset(self, split: str):
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.dataset(split=split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds


class EfficiencyBenchmarkTranslationTask(EfficiencyBenchmarkTask):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        EfficiencyBenchmarkTask.__init__(self, dataset_path, dataset_name, version_override=version_override)


class EfficiencyBenchmarkClassificationTask(EfficiencyBenchmarkTask):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        EfficiencyBenchmarkTask.__init__(self, dataset_path, dataset_name, version_override=version_override)


@dataclass
class EfficiencyBenchmarkInstance:
    input: Union[str, Dict[str, Any]]
    target: str
    id: Optional[str]


def efficiency_benchmark_mt_conversion(
    **kwargs
) -> InstanceConversion:
    def convert(
        instance: Dict[str, Any],
        *,
        input_field: str,
        target_field: str,
        id_field: Optional[str] = None
    ) -> EfficiencyBenchmarkInstance:
        instance = instance["translation"]
        input = get_from_dict(instance, input_field)
        target = get_from_dict(instance, target_field)
        return EfficiencyBenchmarkInstance(
            id=str(get_from_dict(instance, id_field)) if id_field else None,
            input=input,
            target=target
        )
    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)


def efficiency_benchmark_classification_conversion(
    **kwargs,
) -> InstanceConversion:

    def convert(
        instance: Dict[str, Any],
        *,
        label_map: Dict[int, str],
        premise_field: str = "premise",
        hypothesis_field: Optional[str] = "hypothesis",
        label_field: str = "label",
        id_field: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> EfficiencyBenchmarkInstance:
        input = {premise_field: get_from_dict(instance, premise_field)}
        if hypothesis_field is not None:
            input[hypothesis_field] = get_from_dict(instance, hypothesis_field)
        if task_name:
            input["task_name"] = task_name

        label_id = int(get_from_dict(instance, label_field))
        target = label_map[label_id]
        return EfficiencyBenchmarkInstance(
            input=input,
            target=target,
            id=str(get_from_dict(instance, id_field)) if id_field else None
        )

    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)