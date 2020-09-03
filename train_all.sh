#!/bin/bash


## declare an array variable
# declare -a arr=("bigtransfer50x1" 
#                "vgg11" "vgg13" "vgg16" "vgg19"
#                "resnet18" "resnet34" "resnet50" "resnet101" "resnet152"
#                "inceptionv3" "inceptionv4" "inception-resnet-v2"
#                "mobilenet" "mobilenetv2"
#                "seresnet18" "seresnet34" "seresnet50" "seresnet101" "seresnet152"
#                "densenet121" "densenet169" "densenet201"
#                "squeezenet")

declare -a arr=("bigtransfer50x1" 
                "resnet18"
                "densenet121")

## now loop through the above array
for model in "${arr[@]}"
do
    echo "$model"
    printf '%.s*' {1..50}
    echo ""

    trained="0"
    for filename in saved_model/cifar10/*.h5; do
        [ -e "$filename" ] || continue
        file_name="${filename##*/}"
        model_name="${file_name%.*}"
        
        if [ "${model_name,,}" = "${model,,}" ]; then
            trained="1"
            break
        fi
    done
    if [ ${trained} == "0" ]; then
        echo "Need to train for cifar10"
        python main.py --nets=$model --dataset=cifar10 --batch_size=64 --epochs=50 --ops=train
    fi

    trained="0"
    for filename in saved_model/cifar100/*.h5; do
        [ -e "$filename" ] || continue
        file_name="${filename##*/}"
        model_name="${file_name%.*}"
        
        if [ "${model_name,,}" = "${model,,}" ]; then
            trained="1"
            break
        fi
    done
    if [ ${trained} == "0" ]; then
        echo "Need to train for cifar100"
        python main.py --nets=$model --dataset=cifar100 --batch_size=64 --epochs=50 --ops=train
        printf '%.s_' {1..50}
        echo ""
    fi
done
