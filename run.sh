#!/usr/bin/env bash
python -X pycache_prefix=./cache -m src.app.memrl 1
python -X pycache_prefix=./cache -m src.app.memrl 2
python -X pycache_prefix=./cache -m src.app.memrl 3