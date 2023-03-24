"""
Example of wrapping a predictor to read from stdin and write to stdout.

Example usage:
    printf "The movie was good.\nThe movie was bad.\n" > data.txt
    cat data.txt | python stdio_submission.py
"""

from utils import stdio_predictor_wrapper
from typing import Any, Dict, Iterator, Sequence

class GoodBinarySentimentClassifier:
    """
    Simple binary classifier that returns "positive" if "good" or "great" are in the input, otherwise negative.
    """
    def __init__(self):
        self.positive_words = ["good", "great"]

    def predict(
        self,
        inputs: Sequence[Dict[str, Any]]
    ) -> Iterator[str]:

        for input in inputs:
            label = "negative"
            for positive_word in self.positive_words:
                if positive_word in input:
                    label = "positive"
                    break
            yield label


if __name__ == "__main__":
    binary_classifier = GoodBinarySentimentClassifier()
    stdio_predictor_wrapper(binary_classifier)

