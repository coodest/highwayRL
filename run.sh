#!/bin/bash

# remove __pycache__
find . -type d -name __pycache__ -exec rm -r {} \+

# display setup
xhost +"local:docker@" 

# run tensorboard (run in a separate terminal)
# tensorboard --logdir ./output/

# delete output folder
docker run --user=user --volume $(pwd):/work --rm --interactive --tty rl rm -rf ./output

# run code: like for PROFILE in {1..10} 24 25
for INDEX in 2
do
docker run --user=user --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/work --rm --interactive --tty rl python -X pycache_prefix=./cache -m src.app.memrl --env_type atari --index $INDEX
done