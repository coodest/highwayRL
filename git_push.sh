#!/usr/bin/env bash
## 1. set target ci file
#target_file=".gitlab-ci.yml"
#echo "The target file is:"
#echo ${target_file}
#
## 2. switch runner
#resultA=$(grep "G1-A" ${target_file})
#resultB=$(grep "G1-B" ${target_file})
#resultC=$(grep "G1-C" ${target_file})
#resultD=$(grep "G1-D" ${target_file})
#
#if [[ -n "$resultA" ]]; then
#    sed -i 's/- G1-A/- G1-B/g' ${target_file}
#fi
#if [[ -n "$resultB" ]]; then
#    sed -i 's/- G1-B/- G1-D/g' ${target_file}
#fi
#if [[ -n "$resultC" ]]; then
#    sed -i 's/- G1-C/- G1-D/g' ${target_file}
#fi
#if [[ -n "$resultD" ]]; then
#    sed -i 's/- G1-D/- G1-A/g' ${target_file}
#fi

## 3. commit with context information
git config --global credential.helper store
git config --global user.email 350526878@qq.com
git config --global user.name root
git rm -r --cached .
git add .
git --no-pager diff --cached ./
commit_label=$(head -n 1 ./CHANGELOG)
git commit -m "${commit_label}"
git push -u origin master
date
