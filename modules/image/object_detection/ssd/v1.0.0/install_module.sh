#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=ssd-1.0.0.tar.gz

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

tar -zcvf $module_path resources

hub install $module_path

rm $module_path

echo "Successfully create $module_path"
