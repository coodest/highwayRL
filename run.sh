#!/bin/bash

# remove __pycache__
find . -type d -name __pycache__ -exec rm -r {} \+

# display setup
xhost +"local:docker@" 

# run tensorboard (run in a separate terminal)
# tensorboard --logdir ./output/

# clean existing container
docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)

PARALLEL=-t
# PARALLEL=-td

START=0
END=0

# for ENV_TYPE in atari maze toy_text football
for ENV_TYPE in maze
do
    for RUN in $(seq $START $END)
    do
        for ENV in $(sed 1d ./assets/${ENV_TYPE}.txt)
        do  
            if [ ${ENV:0:1} == "#" ]; then
                continue
            fi
            docker run --user=worker --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --shm-size=40gb --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --rm --interactive $PARALLEL meetingdocker/rl:pt_0.2 python -X pycache_prefix=./cache -m src.app.highwayrl --env_type $ENV_TYPE --env_name $ENV --run $RUN
        done
        sleep 1
        while [ `docker ps --all | wc -l` -gt 3 ]
        do 
            sleep 1
        done
    done
    
    while [ `docker ps --all | wc -l` -gt 1 ]
    do 
        sleep 1
    done
    echo "${ENV_TYPE} done"
done

# ./wandb.sh