#!/bin/bash

# remove __pycache__
find . -type d -name __pycache__ -exec rm -r {} \+

# display setup
xhost +"local:docker@" 

# run tensorboard (run in a separate terminal)
# tensorboard --logdir ./output/

# delete output folder
docker run --user=worker --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 rm -rf ./output

# ENV_TYPE=atari
ENV_TYPE=football

# run code: like for PROFILE in {1..10} 24 25
for INDEX in 1
do
docker run --user=worker --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 python -X pycache_prefix=./cache -m src.app.memrl --env_type $ENV_TYPE --index $INDEX
done