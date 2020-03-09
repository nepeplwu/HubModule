#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=ssd_mobilenet_v1_pascal-2.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ssd_mobilenet_v1_voc ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar
    tar xvf ssd_mobilenet_v1_voc.tar
    rm ssd_mobilenet_v1_voc.tar
fi

cd $script_path/

python abandoned_create_module.py

echo "Successfully create $module_path"
