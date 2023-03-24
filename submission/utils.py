import torch
from typing import Optional, Union
import sys
import json
import os


def resolve_device(device: Optional[Union[int, str, torch.device]] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            # TODO (epwalsh, dirkgr): automatically pick which GPU to use when there are multiple
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif isinstance(device, int):
        if device >= 0:
            return torch.device(f"cuda:{device}")
        else:
            return torch.device("cpu")
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(f"unexpected type for 'device': '{device}'")


def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    for line in sys.stdin:
        line = line.rstrip()
        inputs = json.loads(line)
        assert type(inputs) is list
        outputs = predictor.predict(inputs=inputs)
        # Writes are \n deliminated, so adding \n is essential to separate this write from the next loop iteration.
        outputs = [{"output": o} for o in outputs]
        sys.stdout.write(f"{json.dumps(outputs)}\n")
        # Writes to stdout are buffered. The flush ensures the output is immediately sent through the pipe
        # instead of buffered.
        sys.stdout.flush()
