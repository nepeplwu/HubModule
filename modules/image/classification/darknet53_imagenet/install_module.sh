set -v
set -o nounset
set -o errexit

wget --no-check-certificate https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar
tar -xvf DarkNet53_ImageNet1k_pretrained.tar
rm DarkNet53_ImageNet1k_pretrained.tar

cd ..
tar zcvf darknet.tar.gz darknet53_imagenet
hub install darknet.tar.gz

rm -rf darknet.tar.gz darknet/DarkNet53_ImageNet1k_pretrained
