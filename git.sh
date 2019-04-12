#!/usr/bin/env bash

git add .
git commit -m $1
echo "You are updating github repo with commit message $1"
git push
echo "Finished!"
