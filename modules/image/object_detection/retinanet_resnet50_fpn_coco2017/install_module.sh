#!/bin/bash
set -v
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=retinanet_resnet50_fpn_coco2017-2.0.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d faster_rcnn_r50_2x.tar ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar
    tar xvf retinanet_r50_fpn_1x.tar
    rm retinanet_r50_fpn_1x.tar
fi

cd $script_path/

tar -zcvf $module_path resources

hub install $module_path

rm -rf resources/retinanet_r50_fpn_1x $module_path

echo "Successfully create $module_path"
