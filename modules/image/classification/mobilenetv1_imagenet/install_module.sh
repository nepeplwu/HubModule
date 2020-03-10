#!/bin/bash
set -v
set -o nounset
set -o errexit

wget --no-check-certificate http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar -xvf MobileNetV1_pretrained.tar
rm MobileNetV1_pretrained.tar

cd ..
tar -zcvf mobilenet_v1.tar.gz mobilenetv1_imagenet
hub install mobilenet_v1.tar.gz

rm -rf mobilenet_v1.tar.gz mobilenet_v1_imagenet/MobileNetV1_pretrained
