#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=yolov3_darknet53_coco2017-2.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d yolov3_darknet ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar
    tar xvf yolov3_darknet.tar
    rm yolov3_darknet.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
