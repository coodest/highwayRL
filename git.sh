#!/usr/bin/env bash
# commit with context information
echo "push"
git config --global credential.helper store
git config --global user.email 350526878@qq.com
git config --global user.name Heptagram
git config core.filemode false
git rm -r --cached .
git add .
git --no-pager diff --cached ./
commit_label=$(head -n 1 ./CHANGELOG)
git commit -m "${commit_label}"
git push -u origin master

# show time
date

