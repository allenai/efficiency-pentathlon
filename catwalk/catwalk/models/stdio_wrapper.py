import subprocess
import os

from typing import Any, Dict, Iterator, Sequence, List
import more_itertools
from catwalk.models.template import SubmissionTemplate
from tqdm import tqdm
import json

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
        self.model = MockModel()
        self._convert_fn = lambda text: " ".join(text[k] for k in text.keys())
        self._process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _exhaust_and_yield_stdout(self, block_until_read_num_instances: int = None):
        """
        Read everything from the stdout pipe.
        This function uses stdout.readline() to read one prediction at a time.
        stdout.readline() is either blocking or non-blocking (in this case returns "" if nothing is available),
        and the behavior is determined by calling os.set_blocking(self._process.stdout.fileno(), False/True).
        To avoid complicated async/threaded code, we instead toggle the blocking behavior as needed.
        During non-blocking operation we empty the pipe, but don't wait for additional predictions.
        During blocking, we block reads until a certain number of predicitons is read (used to ensure we receive predictions for all instances).

        block_until_read_num_instances: if None then non-blocking. Otherwise, block until this many predictions are read.
        """
        if block_until_read_num_instances is None:
            os.set_blocking(self._process.stdout.fileno(), False)
            block_until_read_num_instances = 1000000000
        else:
            os.set_blocking(self._process.stdout.fileno(), True)

        num_read = 0
        while True and num_read < block_until_read_num_instances:
            # output is bytes, decode to str
            # Also necessary to remove the \n from the end of the label.
            predictions = self._process.stdout.readline().decode("utf-8").rstrip()
            if predictions == "":
                # Nothing in stdout
                break
            else:
                yield predictions
                num_read += 1

    def predict(  # type: ignore
        self,
        *,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[str]:

        num_batches_yielded = 0
        batches = list(more_itertools.chunked(instances, batch_size))
        num_batches = len(batches)
        with tqdm(instances, desc="Processing instances") as instances:

            for batch in batches:
                self._process.stdin.write(f"{json.dumps(batch)}\n".encode("utf-8"))
                self._process.stdin.flush()

                # Check for anything in stdout but don't block sending additional predictions.
                for msg in self._exhaust_and_yield_stdout(None):
                    num_batches_yielded += 1
                    yield msg

            # Now read from stdout until we have hit the required number.
            num_batches_to_read = num_batches - num_batches_yielded
            for msg in self._exhaust_and_yield_stdout(num_batches_to_read):
                yield msg
