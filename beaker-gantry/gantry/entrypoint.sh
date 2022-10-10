#!/bin/bash

set -eo pipefail

# Ensure we have all the environment variables we need.
for env_var in "$GITHUB_TOKEN" "$GITHUB_REPO" "$GIT_REF"; do
    if [[ -z "$env_var" ]]; then
        echo >&2 "error: required environment variable is empty"
        exit 1
    fi
done

# Initialize conda for bash.
# See https://stackoverflow.com/a/58081608/4151392
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo "
##############################################
# [GANTRY] [1/3] Installing prerequisites... #
##############################################
"

# Install GitHub CLI.
conda install gh --channel conda-forge

# Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
gh auth setup-git

echo "
#########################################
# [GANTRY] [2/3] Cloning source code... #
#########################################
"

# Clone the repo and checkout the target commit.
gh repo clone "$GITHUB_REPO" .
git checkout "$GIT_REF"

echo "
###############################################
# [GANTRY] [3/3] Reconstructing Python env... #
###############################################
"

if [[ -z "$VENV_NAME" ]]; then
    VENV_NAME=venv
fi
if [[ -z "$CONDA_ENV_FILE" ]]; then
    # shellcheck disable=SC2296
    CONDA_ENV_FILE="${{ CONDA_ENV_FILE }}"
fi
if [[ -z "$PIP_REQUIREMENTS_FILE" ]]; then
    # shellcheck disable=SC2296
    PIP_REQUIREMENTS_FILE="${{ PIP_REQUIREMENTS_FILE }}"
fi

if conda activate $VENV_NAME &>/dev/null; then
    echo "[GANTRY] Using existing conda environment '$VENV_NAME'"
    # The virtual environment already exists. Possibly update it based on an environment file.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        echo "[GANTRY] Updating environment from conda env file '$CONDA_ENV_FILE'..."
        conda env update -f "$CONDA_ENV_FILE"
    fi
else
    # The virtual environment doesn't exist yet. Create it.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        # Create from the environment file.
        echo "[GANTRY] Initializing environment from conda env file '$CONDA_ENV_FILE'..."
        conda env create -n "$VENV_NAME" -f "$CONDA_ENV_FILE" 
    elif [[ -z "$PYTHON_VERSION" ]]; then
        # Create a new empty environment with the whatever the default Python version is.
        echo "[GANTRY] Initializing environment with default Python version..."
        conda create -n "$VENV_NAME" pip
    else
        # Create a new empty environment with the specific Python version.
        echo "[GANTRY] Initializing environment with Python $PYTHON_VERSION..."
        conda create -n "$VENV_NAME" "python=$PYTHON_VERSION" pip
    fi
    conda activate "$VENV_NAME"
fi

# Install GPU utilities
conda install -c conda-forge gputil

# Check for a 'requirements.txt' and/or 'setup.py' file.
if [[ -f 'setup.py' ]] && [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
    echo "[GANTRY] Installing packages from 'setup.py' and '$PIP_REQUIREMENTS_FILE'..."
    pip install . -r "$PIP_REQUIREMENTS_FILE"
elif [[ -f 'setup.py' ]]; then
    echo "[GANTRY] Installing packages from 'setup.py'..."
    pip install .
elif [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
    echo "[GANTRY] Installing dependencies from '$PIP_REQUIREMENTS_FILE'..."
    pip install -r "$PIP_REQUIREMENTS_FILE"
fi

PYTHONPATH="$(pwd)"
export PYTHONPATH

# Create directory for results.
# shellcheck disable=SC2296
mkdir -p "${{ RESULTS_DIR }}/.gantry"


echo "
#############################
# [GANTRY] Environment info #
#############################
"

echo "Using $(python --version) from $(which python)"
echo "Packages:"
if which sed >/dev/null; then
    pip freeze | sed 's/^/- /'
else
    pip freeze
fi

echo "
#############################
# [GANTRY] Setup complete ✓ #
#############################
"

cat << EOF > execute.py
import subprocess as sp
import sys
import os
from threading import Timer
import csv

def get_gpu_status():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --id=0 --query-gpu=timestamp,memory.used,power.draw --format=csv"
    try:
        gpu_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    gpu_info = gpu_info[0].split(",")
    return gpu_info


def write_gpu_status(writer, fout):
    gpu_info = get_gpu_status()
    writer.writerow(gpu_info)
    fout.flush()

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args,**self.kwargs)
            # print(' ')

if __name__ == "__main__":
    print(sys.argv)
    interval = 0.1
    header = ["Timestamp", "Memory", "Power"]
    fout = open("log.csv", "w", encoding="UTF8", newline="")
    writer = csv.writer(fout)
    writer.writerow(header)
    timer = RepeatTimer(interval, write_gpu_status, [writer, fout])
    timer.start()
    print(" ".join(sys.argv[1:]))
    os.system(" ".join(sys.argv[1:]))
    timer.cancel()
    fout.close()

    max_mem, power = 0, 0
    with open("log.csv", "r", encoding="UTF8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            m, p = row["Memory"], row["Power"]
            m = float(m.split()[0])
            if m < 10:
                continue
            p = float(p.split()[0])
            max_mem = max(max_mem, m)
            power = power + interval * p
    print(f"Memory: {max_mem:.2f} MiB| Power: {power:.2f} W.s")
EOF

python rprof/profile_gpu.py & 
RPROF_PID=$!

sh profile_cpu.sh &
CPU_PID=$!
while true; do docker stats $CONTAINER_ID --no-stream --format "{{.CPUPerc}} {{.MemPerc}}" | tee --append stats.txt; sleep 1; done
# Execute the arguments to this script as commands themselves, piping output into a log file.
# shellcheck disable=SC2296
exec "$@" 2>&1 | tee "${{ RESULTS_DIR }}/.gantry/out.log"
# python execute.py "$@" 2>&1 | tee "${{ RESULTS_DIR }}/.gantry/out.log"
