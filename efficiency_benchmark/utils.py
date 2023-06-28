from collections import abc
from typing import Callable, List, Sequence


class MappedSequence(abc.Sequence):
    """
    Produces a sequence that applies a function to every element of another sequence.

    This is similar to Python's :func:`map`, but it returns a sequence instead of a :class:`map` object.

    :param fn: the function to apply to every element of the inner sequence. The function should take
               one argument.
    :param inner_sequence: the inner sequence to map over

    From https://github.com/allenai/tango/blob/main/tango/common/sequences.py#L176
    """

    def __init__(self, fn: Callable, inner_sequence: Sequence):
        self.inner = inner_sequence
        self.fn = fn

    def __getitem__(self, item):
        if isinstance(item, slice):
            new_inner = None
            try:
                # special case for a special library
                from datasets import Dataset

                if isinstance(self.inner, Dataset):
                    new_inner = self.inner.select(range(*item.indices(len(self.inner))))
            except ImportError:
                pass
            if new_inner is None:
                new_inner = self.inner[item]
            return MappedSequence(self.fn, new_inner)
        else:
            item = self.inner.__getitem__(item)
            return self.fn(item)

    def __len__(self):
        return self.inner.__len__()

    def __contains__(self, item):
        return any(e == item for e in self)


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """
    Transforms the potential gpu_ids string into a list of int values.
    From https://github.com/mlco2/codecarbon/blob/master/codecarbon/core/config.py

    Args:
        gpu_ids_str (str): The config file or environment variable value for `gpu_ids`
        which is read as a string and should be parsed into a list of ints

    Returns:
        list[int]: The list of GPU ids available declared by the user.
            Potentially empty.
    """
    if not isinstance(gpu_ids_str, str):
        return gpu_ids_str

    gpu_ids_str = "".join(c for c in gpu_ids_str if (c.isalnum() or c == ","))
    str_ids = [gpu_id for gpu_id in gpu_ids_str.split(",") if gpu_id]
    return list(map(int, str_ids))