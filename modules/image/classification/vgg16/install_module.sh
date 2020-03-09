wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar
tar -xvf VGG16_pretrained.tar
rm VGG16_pretrained.tar
cd ..
tar zcvf vgg16.tar.gz vgg16
hub install vgg16.tar.gz
rm vgg16.tar.gz
rm -r vgg16/VGG16_pretrained
