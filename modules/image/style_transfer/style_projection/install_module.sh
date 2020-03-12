#!/bin/bash
set -v
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=style_projection_coco_wikiart-1.0.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d style_projection_enc.tar ]
then
    wget --no-check-certificate https://bj.bcebos.com/paddlehub/style_transfer/style_projection_enc.tar
    tar xvf style_projection_enc.tar
    rm style_projection_enc.tar
fi

if [ ! -d style_projection_dec.tar ]
then
    wget --no-check-certificate https://bj.bcebos.com/paddlehub/style_transfer/style_projection_dec.tar
    tar xvf style_projection_dec.tar
    rm style_projection_dec.tar
fi

cd $script_path/

tar -zcvf $module_path resources

hub install $module_path

rm -rf resources/style_projection_dec resources/style_projection_enc $module_path

echo "Successfully create $module_path"
