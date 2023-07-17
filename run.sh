#!/bin/bash

# remove __pycache__
find . -type d -name __pycache__ -exec rm -r {} \+

# display setup
xhost +"local:docker@" 

# run tensorboard (run in a separate terminal)
# docker run --runtime=nvidia --gpus all --volume $(pwd):/home --rm --publish 6006:6006 rl rm -rf ./output && tensorboard --logdir ./output/

# profile ID
# PROFILE=1
# PROFILE=2
# PROFILE=3
# PROFILE=4
# PROFILE=5
# PROFILE=6
# PROFILE=7
# PROFILE=8
# PROFILE=9
# PROFILE=10
# PROFILE=11
# PROFILE=12
# PROFILE=13
# PROFILE=14
# PROFILE=15
# PROFILE=16
# PROFILE=17
# PROFILE=18
# PROFILE=19
# PROFILE=20
# PROFILE=21
# PROFILE=22
# PROFILE=23
PROFILE=24
# PROFILE=25
# PROFILE=26

# delete output folder
docker run --user=user --volume $(pwd):/work --rm --interactive --tty rl rm -rf ./output

# run code
docker run --user=user --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/work --rm --interactive --tty rl python -X pycache_prefix=./cache -m src.app.memrl $PROFILE