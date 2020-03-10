#!/bin/bash
set -v
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=yolov3_mobilenet_v1_coco2017-2.0.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ssd_mobilenet_v1_voc ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
    tar xvf yolov3_mobilenet_v1.tar
    rm yolov3_mobilenet_v1.tar
fi

cd $script_path/

tar -zcvf $module_path resources

hub install $module_path

rm -rf resources/yolov3_mobilenet_v1 $module_path

echo "Successfully create $module_path"
