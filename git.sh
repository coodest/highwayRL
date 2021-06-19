#!/usr/bin/env bash
# 1. commit with context information
git config --global credential.helper store
git config --global user.email 350526878@qq.com
git config --global user.name Heptagram
git rm -r --cached .
git add .
git update-index --chmod=+x run.sh
git update-index --chmod=+x git.sh
git --no-pager diff --cached ./
commit_label=$(head -n 1 ./CHANGELOG)
git commit -m "${commit_label}"
git push -u origin master

# 2. show time
date

