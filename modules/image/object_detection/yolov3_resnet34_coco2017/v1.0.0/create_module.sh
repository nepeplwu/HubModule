#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=yolov3_resnet34_coco2017-1.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d yolov3_r34 ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
    tar xvf yolov3_r34.tar
    rm yolov3_r34.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
