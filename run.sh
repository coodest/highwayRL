#!/bin/bash

# Configure the resources required
# v100 partition (for GPU jobs) notes:
# max job time 2-00:00:00
# max cpus/node: 32
# max RAM per cpu: 4915 MB
# max RAM per node 153 GB
# cost per node: 0.5 SU per cpu-hour + 33 SU per GPU-hour
#SBATCH -p v100
#SBATCH -N 1  # number of nodes
#SBATCH -n 24  # number of cores
#SBATCH --time=0-00:20:00  # time of execution, D-HH:MM:SS
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2

# Configure the notification
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=a1226115@adelaide.edu.au

# ---------Execute the script by the following---------------
# pip install pipreqs
# pipreqs .

rm -rf ./output
pkill -f tensorboard
tensorboard --logdir ./output/summary/ &
sleep 1

# python -X pycache_prefix=./cache -m src.app.memrl 1
# python -X pycache_prefix=./cache -m src.app.memrl 2
# python -X pycache_prefix=./cache -m src.app.memrl 3
# python -X pycache_prefix=./cache -m src.app.memrl 4
# python -X pycache_prefix=./cache -m src.app.memrl 5
# python -X pycache_prefix=./cache -m src.app.memrl 6
# python -X pycache_prefix=./cache -m src.app.memrl 7
# python -X pycache_prefix=./cache -m src.app.memrl 8
# python -X pycache_prefix=./cache -m src.app.memrl 9
# python -X pycache_prefix=./cache -m src.app.memrl 10
# python -X pycache_prefix=./cache -m src.app.memrl 11
# python -X pycache_prefix=./cache -m src.app.memrl 12
# python -X pycache_prefix=./cache -m src.app.memrl 13
# python -X pycache_prefix=./cache -m src.app.memrl 14
# python -X pycache_prefix=./cache -m src.app.memrl 15
# python -X pycache_prefix=./cache -m src.app.memrl 16
# python -X pycache_prefix=./cache -m src.app.memrl 17
# python -X pycache_prefix=./cache -m src.app.memrl 18
# python -X pycache_prefix=./cache -m src.app.memrl 19
# python -X pycache_prefix=./cache -m src.app.memrl 20
# python -X pycache_prefix=./cache -m src.app.memrl 21
# python -X pycache_prefix=./cache -m src.app.memrl 22
# python -X pycache_prefix=./cache -m src.app.memrl 23
python -X pycache_prefix=./cache -m src.app.memrl 24
# python -X pycache_prefix=./cache -m src.app.memrl 25
# python -X pycache_prefix=./cache -m src.app.memrl 26

# python -X pycache_prefix=./cache -m src.app.memrl 27
# python -X pycache_prefix=./cache -m src.app.memrl 28
# python -X pycache_prefix=./cache -m src.app.memrl 29
# python -X pycache_prefix=./cache -m src.app.memrl 30
# python -X pycache_prefix=./cache -m src.app.memrl 31
# python -X pycache_prefix=./cache -m src.app.memrl 32
# python -X pycache_prefix=./cache -m src.app.memrl 33
# python -X pycache_prefix=./cache -m src.app.memrl 34
# python -X pycache_prefix=./cache -m src.app.memrl 35
# python -X pycache_prefix=./cache -m src.app.memrl 36
# python -X pycache_prefix=./cache -m src.app.memrl 37
# python -X pycache_prefix=./cache -m src.app.memrl 38
# python -X pycache_prefix=./cache -m src.app.memrl 39
# python -X pycache_prefix=./cache -m src.app.memrl 40
# python -X pycache_prefix=./cache -m src.app.memrl 41
# python -X pycache_prefix=./cache -m src.app.memrl 42
# python -X pycache_prefix=./cache -m src.app.memrl 43
# python -X pycache_prefix=./cache -m src.app.memrl 44
# python -X pycache_prefix=./cache -m src.app.memrl 45
# python -X pycache_prefix=./cache -m src.app.memrl 46
# python -X pycache_prefix=./cache -m src.app.memrl 47
# python -X pycache_prefix=./cache -m src.app.memrl 48
# python -X pycache_prefix=./cache -m src.app.memrl 49
# python -X pycache_prefix=./cache -m src.app.memrl 50
# python -X pycache_prefix=./cache -m src.app.memrl 51
# python -X pycache_prefix=./cache -m src.app.memrl 52
# python -X pycache_prefix=./cache -m src.app.memrl 53
# python -X pycache_prefix=./cache -m src.app.memrl 54
# python -X pycache_prefix=./cache -m src.app.memrl 55
# python -X pycache_prefix=./cache -m src.app.memrl 56
# python -X pycache_prefix=./cache -m src.app.memrl 57

