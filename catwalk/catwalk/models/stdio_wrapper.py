import json
import os
import subprocess
from typing import Any, Dict, Iterator, List, Sequence

import more_itertools

import docker
from catwalk.model import Model


class StdioWrapper(Model):
    """
    A model that wraps a binary that reads from stdin and writes to stdout.
    """

    def __init__(self, cmd: List[str]):
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
            try:
                output_batch = json.loads(output_batch)
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
        line = self._process.stdout.readline().decode("utf-8").rstrip()
        if line == "":
            raise ValueError
        return line

    def predict(  # type: ignore
        self,
        *,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[str]:

        num_batches_yielded = 0
        batches = list(more_itertools.chunked(instances, batch_size))
        num_batches = len(batches)
        for batch in batches:
            self._write_batch(batch)

            # Check for anything in stdout but don't block sending additional predictions.
            output_batches = self._exhaust_and_yield_stdout(None)
            for output_batch in output_batches:
                num_batches_yielded += 1
                for output in output_batch:
                    yield output

        # Now read from stdout until we have hit the required number.
        num_batches_to_read = num_batches - num_batches_yielded
        for output_batch in self._exhaust_and_yield_stdout(num_batches_to_read):
            for output in output_batch:
                yield output

    def start(self, dummy_inputs: List[Dict[str, Any]]) -> List[str]:
        # TODO
        self._process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._write_batch(dummy_inputs)
        dummy_outputs = self._exhaust_and_yield_stdout(1)
        return list(dummy_outputs)

    def stop(self):
        pass


class DockerStdioWrapper(StdioWrapper):
    """
    A model that wraps a binary that reads from stdin and writes to stdout.
    """

    def __init__(self, cmd: List[str]):
        """
        binary_cmd: the command to start the inference binary
        """
        super().__init__(cmd)

    def _set_blocking(self, block_until_read_num_batches: int = None) -> None:
        blocking = block_until_read_num_batches is not None
        self._socket._sock.setblocking(blocking)

    def _write_batch(self, batch: Sequence[Dict[str, Any]]) -> None:
        self._socket._sock.send(f"{json.dumps(batch)}\n".encode("utf-8"))
        self._socket.flush()
    
    def _read_batch(self) -> str:
        try:
            return self._socket.readline()[8:].decode("utf-8").rstrip()
        except:
            raise ValueError

    def start(self, dummy_inputs: List[Dict[str, Any]]) -> List[str]:
        client = docker.DockerClient()
        self._container = client.containers.run(
            image="transformers:latest",
            command=self._cmd,
            name="transformers",
            auto_remove=True,
            remove=True,
            stdin_open=True,
            detach=True,
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=["0"],  # TODO
                    capabilities=[["gpu"]]
                )
            ]
        )
        self._socket = self._container.attach_socket(
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream":1}
        )
        self._write_batch(dummy_inputs)
        dummy_outputs = self._exhaust_and_yield_stdout(1)
        return list(dummy_outputs)

    def stop(self) -> None:
        try:
            self._container.stop()
            self._container.remove()
        except:
            pass
