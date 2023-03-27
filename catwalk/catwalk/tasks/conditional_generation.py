import functools
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
import datasets
from tango.common.sequences import MappedSequence

from catwalk.task import Task, InstanceConversion
from catwalk.tasks.huggingface import get_from_dict


class ConditionalGenerationTask(Task):
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


class MachineTranslationTask(ConditionalGenerationTask):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        ConditionalGenerationTask.__init__(self, dataset_path, dataset_name, version_override=version_override)


@dataclass
class ConditionalGenerationInstance:
    id: Optional[str]
    source: str
    target: str


def conditional_generation_convert(
    instance: Dict[str, Any],
    *,
    source_field: str,
    target_field: str,
    id_field: Optional[str] = None
) -> ConditionalGenerationInstance:
    instance = instance["translation"]
    source = get_from_dict(instance, source_field)
    target = get_from_dict(instance, target_field)
    return ConditionalGenerationInstance(
        id=str(get_from_dict(instance, id_field)) if id_field else None,
        source=source,
        target=target
    )


def conditional_generation_conversion(
    **kwargs
) -> InstanceConversion:
    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(conditional_generation_convert, **kwargs)
