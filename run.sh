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

# python -X pycache_prefix=./cache -m src.app.memrl 1
# python -X pycache_prefix=./cache -m src.app.memrl 2
# python -X pycache_prefix=./cache -m src.app.memrl 3
# python -X pycache_prefix=./cache -m src.app.memrl 4
# python -X pycache_prefix=./cache -m src.app.memrl 5
python -X pycache_prefix=./cache -m src.app.memrl 6
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
# python -X pycache_prefix=./cache -m src.app.memrl 24
# python -X pycache_prefix=./cache -m src.app.memrl 25
# python -X pycache_prefix=./cache -m src.app.memrl 26
