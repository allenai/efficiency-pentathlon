#!/bin/bash

set -eo pipefail

# beaker image pull haop/submission submission
# echo "Pulled submission docker image"

echo "Running submission docker"
echo docker run --env-file ./.efficiency-benchmark-env-list -t submission exec "$@"
docker run --env-file ./.efficiency-benchmark-env-list -t submission exec "$@"
