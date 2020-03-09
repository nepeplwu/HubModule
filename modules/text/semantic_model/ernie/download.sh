wget https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz --no-check-certificate
tar xzvf ERNIE_1.0_max-len-512.tar.gz
mkdir assets
mv ernie_config.json params/ vocab.txt assets/
