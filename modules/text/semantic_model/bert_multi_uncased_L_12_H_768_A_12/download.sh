wget https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz --no-check-certificate
tar xvf multilingual_L-12_H-768_A-12.tar.gz
mv multilingual_L-12_H-768_A-12 assets

cd ..
python /qjx/PaddleHub/paddlehub/commands/hub.py install bert_multi_uncased_L_12_H_768_A_12
cd bert_multi_uncased_L_12_H_768_A_12
python test_embedding.py
