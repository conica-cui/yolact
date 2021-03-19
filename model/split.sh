#!/bin/bash

for d in */ ; do
    echo "$d"
    cd "$d"
    echo "Split x* files"
    split -b 90000k yolact_plus_base.pth
    rm yolact_plus_base.pth
    cd -
done
