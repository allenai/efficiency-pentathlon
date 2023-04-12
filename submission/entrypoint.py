import argparse
import json
import sys

import transformers
from mbart import MBART
from t5 import T5
from example_stdio_submission_sst import GoodBinarySentimentClassifier


# We provide this
def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    for line in sys.stdin:
        line = line.rstrip()
        inputs = json.loads(line)
        assert isinstance(inputs, list)
        # Participants need to connect their inference code to our wrapper through the following line.
        outputs = predictor.predict(inputs=inputs)
        # Writes are \n deliminated, so adding \n is essential to separate this write from the next loop iteration.
        outputs = [o for o in outputs]
        sys.stdout.write(f"{json.dumps(outputs)}\n")
        # Writes to stdout are buffered. The flush ensures the output is immediately sent through the pipe
        # instead of buffered.
        sys.stdout.flush()


if __name__ == "__main__":
    # We read outputs from stdout, and it is crucial to surpress unnecessary logging to stdout
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    transformers.logging.disable_progress_bar()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    if "t5" in args.model:
        predictor = T5(
            pretrained_model_name_or_path=args.model,
            task=args.task
        )
    elif args.model == "mbart":
        predictor = MBART()
    elif args.model == "debug":
        predictor = GoodBinarySentimentClassifier()
    else:
        raise NotImplementedError()
    stdio_predictor_wrapper(predictor)
