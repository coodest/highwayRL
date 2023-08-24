#!/bin/bash

# remove __pycache__
find . -type d -name __pycache__ -exec rm -r {} \+

rm -rf ./wandb

# display setup
xhost +"local:docker@" 

# run tensorboard (run in a separate terminal)
# tensorboard --logdir ./output/

# delete output folder
docker run --user=worker --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 rm -rf ./output

ENV_TYPE=maze
# ENV_TYPE=toy_text
# ENV_TYPE=football  
# ENV_TYPE=atari

for ENV in $(sed 1d ./assets/${ENV_TYPE}.txt)
do
    for RUN in 0
    do
        # tty
        docker run --user=worker --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive --tty meetingdocker/rl:pt_0.1 python -X pycache_prefix=./cache -m src.app.memrl --env_type $ENV_TYPE --env_name $ENV --run $RUN
        # detach
        # docker run --user=worker --runtime=nvidia --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive --detach meetingdocker/rl:pt_0.1 python -X pycache_prefix=./cache -m src.app.memrl --env_type $ENV_TYPE --env_name $ENV --run $RUN
    done
done

# docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)