wget --no-check-certificate http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar -xvf MobileNetV1_pretrained.tar
rm  MobileNetV1_pretrained.tar
cd ..
tar -zcvf mobilenet_v1.tar.gz mobilenet_v1
hub install mobilenet_v1.tar.gz
rm mobilenet_v1.tar.gz
rm -r mobilenet_v1/MobileNetV1_pretrained
