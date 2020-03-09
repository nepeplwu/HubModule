#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=retinanet_resnet50_fpn_coco2017-2.0.0.phm

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d retinanet_r50_fpn_1x ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar
    tar xvf retinanet_r50_fpn_1x.tar
    rm retinanet_r50_fpn_1x.tar
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
