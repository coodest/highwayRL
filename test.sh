#!/usr/bin/env bash

# display setup
xhost +"local:docker@" 

# delete output folder
docker run --user=worker --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 rm -rf ./output

# run code
docker run --user=worker --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 python -X pycache_prefix=./cache -m test.any --env_type atari --index 1
