#!/usr/bin/env bash

# display setup
xhost +"local:docker@" 

PARALLEL=--tty

# for ENV_TYPE in maze toy_text football atari
for ENV_TYPE in toy_text
do
    for ENV in $(sed 1d ./assets/${ENV_TYPE}.txt)
    do
        for RUN in 1
        do
            docker run --user=worker --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --shm-size=40gb --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive $PARALLEL meetingdocker/rl:pt_0.1 python -X pycache_prefix=./cache -m test.any --env_type $ENV_TYPE --env_name $ENV --run $RUN --keep_dir
        done
        break
    done
    while [ `docker ps --all | wc -l` -gt 1 ]
    do 
        sleep 1
    done
    echo "${ENV_TYPE} done"
    break
done