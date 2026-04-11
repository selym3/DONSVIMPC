# DONSVIMPC

## Installation

With uv.

Setup virtual env

```
uv venv
source .venv/bin/activate
```

With uv (for cpu):

```
uv pip install -r robot_planning/requirements.txt
uv pip install -e ./og
uv pip install -e ./ # Installs robot_planning package
```

With uv (for gpu):

```

```

### Fixes

Make a directory to store generated offline dataset (otherwise `mkdir` fails):

```bash
# From the project root
mkdir robot_planning/experiments
```

**TODO: NEED TO BITE THE BULLET AND UPDATE THE PROJECT TO USE UTKU'S GPU**

## Running

To run the various environments in scripts, you must be in the scripts folder, otherwise the config will not load correctly. For example, to run the quadrotor environment:

```sh
cd robot_planning/scripts
python3 run_quadrotor2d.py
```
