wget https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz --no-check-certificate
tar xvf cased_L-24_H-1024_A-16.tar.gz
mv cased_L-24_H-1024_A-16 assets

cd ..
python /qjx/PaddleHub/paddlehub/commands/hub.py install bert_cased_L_24_H_1024_A_16
cd bert_cased_L_24_H_1024_A_16
python test_embedding.py
