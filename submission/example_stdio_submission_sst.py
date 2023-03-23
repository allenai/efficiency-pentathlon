"""
Example of wrapping a predictor to read from stdin and write to stdout.

Example usage:
    printf "The movie was good.\nThe movie was bad.\n" > data.txt
    cat data.txt | python stdio_submission.py
"""

import sys

class GoodBinarySentimentClassifier:
    """
    Simple binary classifier that returns "positive" if "good" or "great" are in the input, otherwise negative.
    """
    def __init__(self):
        self.positive_words = ["good", "great"]

    def predict(self, x: str) -> str:
        label = "negative"

        for positive_word in self.positive_words:
            if positive_word in x:
                label = "positive"
                break

        return label


def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    for line in sys.stdin:
        label = predictor.predict(line)
        # Writes are \n deliminated, so adding \n is essential to separate this write from the next loop iteration.
        sys.stdout.write(f"{label}\n")
        # Writes to stdout are buffered. The flush ensures the output is immediately sent through the pipe
        # instead of buffered.
        sys.stdout.flush()


if __name__ == "__main__":
    binary_classifier = GoodBinarySentimentClassifier()
    stdio_predictor_wrapper(binary_classifier)

