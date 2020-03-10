#!/bin/bash
set -o nounset
set -o errexit

wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar
tar -xvf ResNet50_vd_pretrained.tar
rm ResNet50_vd_pretrained.tar

wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_pretrained.tar
tar -xvf ResNet34_vd_pretrained.tar
rm ResNet34_vd_pretrained.tar

cd ..
tar -zcvf resnet.tar.gz resnet
hub install resnet.tar.gz

rm -rf resnet.tar.gz
rm -rf resnet/ResNet34_vd_pretrained resnet/ResNet50_vd_pretrained
