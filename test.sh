#!/usr/bin/env bash

# display setup
xhost +"local:docker@" 

# run code
docker run --user=user --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/work --rm --interactive --tty rl python -X pycache_prefix=./cache -m test.any 1
