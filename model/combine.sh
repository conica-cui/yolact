#!/bin/bash

for d in */ ; do
    echo "$d"
    cd "$d"
    echo "Combine x* files"
    cat x* > yolact_plus_base.pth
    rm x*
    cd -
done
