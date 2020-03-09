wget https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz --no-check-certificate
tar xzvf ERNIE_Large_en_stable-2.0.0.tar.gz
mkdir assets
mv ernie_config.json params/ vocab.txt assets/
