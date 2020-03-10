#!/bin/bash
set -v
set -o nounset
set -o errexit

wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar
tar -xvf VGG16_pretrained.tar
rm VGG16_pretrained.tar

cd ..
tar zcvf vgg16.tar.gz vgg16_imagenet
hub install vgg16.tar.gz

rm -rf vgg16.tar.gz vgg16/VGG16_pretrained
