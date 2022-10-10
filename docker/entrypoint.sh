#!/bin/bash

set -eo pipefail
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate efficiency


