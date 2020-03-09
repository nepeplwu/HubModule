#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=faster_rcnn_resnet50_coco2017-1.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d faster_rcnn_r50_2x ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar
    tar xvf faster_rcnn_r50_2x.tar
    rm faster_rcnn_r50_2x.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
