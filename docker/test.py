import pyRAPL
pyRAPL.setup()

import time

with pyRAPL.Measurement('bar'):
    for i in range(10):
        time.sleep(1)



