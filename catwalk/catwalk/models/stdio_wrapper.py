from typing import Any, Dict, Iterator, Sequence, List

import more_itertools
import torch
from tango.common import Tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tango.integrations.torch.util import resolve_device

from catwalk.models.template import SubmissionTemplate

import subprocess
import os


"""
import subprocess
import os

p = subprocess.Popen(["python", "submission/example_stdio_submission_sst.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
os.set_blocking(p.stdout.fileno(), False)

#p = subprocess.Popen(["python", "submission/example_stdio_submission_sst.py"], stdin=subprocess.PIPE)


p.stdin.write("movie good.\n".encode("utf-8"))
p.stdin.flush()

p.stdout.flush()
p.stdout.readline()
"""

# This is a hack to work around pytorch specific hard coded assumptions.
class MockModel:
    def eval(self):
        pass

    def num_parameters(self):
        return 0


class StdioWrapper(SubmissionTemplate):
    """
    A model that wraps a binary that reads from stdin and writes to stdout.
    """

    def __init__(self, binary_cmd: List[str]):
        """
        binary_cmd: the command to start the inference binary
        """
        SubmissionTemplate.__init__(self)
        self._cmd = binary_cmd
        self._instance_labels = []
        self._yielded_label_index = -1
        self.model = MockModel()


    def load_model(self):
        self._convert_fn = lambda text: " ".join(text[k] for k in text.keys())
        # start the subprocess
        self._process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # The next line sets stdout to non-blocking. After setting, self._process.stdout.readline() returns
        # '' if there is nothing in the pipe. Otherwise, self._process.stdout.readline blocks until something
        # is available.  This is necessary as we want to always check stdout and yield the output as soon
        # as it's available, but not block additional input to stdin.  Setting this also removes the
        # need for async or threaded code, which reduces complexity of this wrapper significantly.
        os.set_blocking(self._process.stdout.fileno(), False)


    def _exhaust_and_yield_stdout(self):
        while True:
            # output is bytes, decode to str
            # Also necessary to remove the \n from the end of the label.
            prediction = self._process.stdout.readline().decode("utf-8").rstrip()
            if prediction == "":
                # Nothing in stdout
                break
            else:
                self._yielded_label_index += 1
                instance_label = self._instance_labels[self._yielded_label_index]
                yield {
                    "label": instance_label,
                    "prediction": prediction,
                    "acc": (prediction, instance_label),
                }


    def predict(  # type: ignore
        self,
        *,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:

        for instance in instances:
            instance_text = self._convert_fn(instance.text)
            self._process.stdin.write(f"{instance_text}\n".encode("utf-8"))
            self._process.stdin.flush()
            self._instance_labels.append(instance.label)

            # Check for anything in stdout. If it's present, then yield it in the generator.
            for msg in self._exhaust_and_yield_stdout():
                yield msg

