#!/usr/bin/env bash
# 1. git puu
echo "pull"
git pull --rebase

# 2. commit with context information
echo "push"
git config --global credential.helper store
git config --global user.email 350526878@qq.com
git config --global user.name Heptagram
git rm -r --cached .
git add .
git update-index --chmod=+x run.sh
git update-index --chmod=+x git.sh
git update-index --chmod=+x debug.sh
git update-index --chmod=+x submit.sh
git --no-pager diff --cached ./
commit_label=$(head -n 1 ./CHANGELOG)
git commit -m "${commit_label}"
git push -u origin master

# 3. show time
date

