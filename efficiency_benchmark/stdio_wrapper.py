import json
import os
import subprocess
from abc import ABC
from typing import Any, Dict, Iterator, List, Sequence
import tqdm
import more_itertools


class StdioWrapper(ABC):
    """
    A model that wraps a binary that reads from stdin and writes to stdout.
    """

    def __init__(self, *, cmd: List[str]):
        """
        binary_cmd: the command to start the inference binary
        """
        self._cmd = cmd

    def _exhaust_and_yield_stdout(self, block_until_read_num_batches: int = None):
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
        self._set_blocking(block_until_read_num_batches)
        if block_until_read_num_batches is None:
            block_until_read_num_batches = 1000000000

        num_batches_yielded = 0
        while num_batches_yielded < block_until_read_num_batches:
            # output is bytes, decode to str
            # Also necessary to remove the \n from the end of the label.
            try:
                output_batch = self._read_batch()
            except ValueError:
                # Nothing in stdout
                break

            # Very annoyingly, docker socket sometimes attach the output with an 8-byte header,
            # and sometimes not. 
            try:
                output_batch = json.loads(output_batch[8:].decode("utf-8").strip())
            except:
                try:
                    output_batch = json.loads(output_batch.decode("utf-8").strip())
                except:
                    # Irrelavent output in stdout
                    continue
            yield output_batch
            num_batches_yielded += 1

    def _set_blocking(self, block_until_read_num_batches: int = None):
        blocking = block_until_read_num_batches is not None
        os.set_blocking(self._process.stdout.fileno(), blocking)

    def _write_batch(self, batch: Sequence[Dict[str, Any]]) -> None:
        self._process.stdin.write(f"{json.dumps(batch)}\n".encode("utf-8"))
        self._process.stdin.flush()
    
    def _read_batch(self) -> str:
        line = self._process.stdout.readline()
        if line.decode("utf-8").strip() == "":
            raise ValueError
        return line

    def predict(  # type: ignore
        self,
        *,
        input_batches: List[List[str]],
        max_batch_size: int
    ) -> Iterator[str]:
        for input_batch in tqdm.tqdm(input_batches, desc="Making predictions", miniters=10):

            # Make sure the batch size does not exceed a user defined maximum.
            # Split into smaller batches if necessary.
            splitted_batches = list(more_itertools.chunked(input_batch, max_batch_size))
            num_splitted_batches = len(splitted_batches)
            num_batches_yielded = 0
            for batch in splitted_batches:
                self._write_batch(batch)
                # Feed all splitted batches without blocking.
                output_batches = self._exhaust_and_yield_stdout(None)
                for output_batch in output_batches:
                    num_batches_yielded += 1
                    for output in output_batch:
                        yield output

            # Now read from stdout until we have hit the required number.
            # Legacy code from non-blocking mode.
            num_batches_to_read = num_splitted_batches - num_batches_yielded
            if num_batches_to_read > 0:
                for output_batch in self._exhaust_and_yield_stdout(num_batches_to_read):
                    for output in output_batch:
                        yield output

    def provide_offline_configs(
            self,
            offline_data_path: str,
            offline_output_file: str,
            limit: int = -1
    ) -> bool:
        configs = {
            "offline_data_path": offline_data_path,
            "offline_output_path": offline_output_file,
            "limit": limit
        }
        os.set_blocking(self._process.stdout.fileno(), True)
        self._process.stdin.write(f"{json.dumps(configs)}\n".encode("utf-8"))
        self._process.stdin.flush()

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Model and data loaded. Start the timer.":
                break

    def block_for_prediction(
            self
    ) -> bool:
        os.set_blocking(self._process.stdout.fileno(), True)

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Offiline prediction done. Stop the timer.":
                break
    
    def block_for_outputs(
            self
    ) -> bool:
        os.set_blocking(self._process.stdout.fileno(), True)

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Offiline outputs written. Exit.":
                break

    def start(self):
        self._process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def dummy_predict(self, dummy_inputs: List[Dict[str, Any]]) -> List[str]:
        self._write_batch(dummy_inputs)
        dummy_outputs = self._exhaust_and_yield_stdout(1)
        return list(dummy_outputs)
    
    def stop(self):
        pass
