#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=faster_rcnn_resnet50_coco2017-1.1.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d faster_rcnn_r50_2x.tar ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar
    tar xvf faster_rcnn_r50_2x.tar
    rm faster_rcnn_r50_2x.tar
fi

cd $script_path/

tar -zcvf $module_path resources

hub install $module_path

rm -rf resources/faster_rcnn_r50_2x $module_path

echo "Successfully create $module_path"
