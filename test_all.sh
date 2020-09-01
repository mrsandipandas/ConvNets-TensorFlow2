#!/bin/bash

for filename in saved_model/cifar10/*.h5; do
    [ -e "$filename" ] || continue
    file_name="${filename##*/}"
    model="${file_name%.*}"
    python main.py --nets=$model --dataset=cifar10 --ops=test
    sleep 5
done

for filename in saved_model/cifar100/*.h5; do
    [ -e "$filename" ] || continue
    file_name="${filename##*/}"
    model="${file_name%.*}"
    python main.py --nets=$model --dataset=cifar100 --ops=test
    sleep 5
done