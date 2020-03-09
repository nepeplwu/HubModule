#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=ssd_vgg16_300_coco2017-2.0.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ssd_vgg16_300.tar ]
then
    wget --no-check-certificate https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar
    tar xvf ssd_vgg16_300.tar
    rm ssd_vgg16_300.tar
fi

cd $script_path/

tar -zcvf $module_path resources

hub install $module_path

rm -rf resources/ssd_vgg16_300 $module_path

echo "Successfully create $module_path"
