#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=yolov3_mobilenet_v1_coco2017-1.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d yolov3_mobilenet_v1 ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
    tar xvf yolov3_mobilenet_v1.tar
    rm yolov3_mobilenet_v1.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
