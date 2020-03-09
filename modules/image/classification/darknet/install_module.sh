wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar
tar -xvf DarkNet53_ImageNet1k_pretrained.tar
rm DarkNet53_ImageNet1k_pretrained.tar
cd ..
tar zcvf darknet.tar.gz darknet
hub install darknet.tar.gz
rm -r darknet/DarkNet53_ImageNet1k_pretrained
