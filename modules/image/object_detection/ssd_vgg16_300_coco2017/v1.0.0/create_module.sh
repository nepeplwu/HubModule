#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=ssd_vgg16_300_coco2017-1.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ssd_vgg16_300 ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar
    tar xvf ssd_vgg16_300.tar
    rm ssd_vgg16_300.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
