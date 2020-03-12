wget https://paddlehub.bj.bcebos.com/model/nlp/chinese_rbt3_L-3_H-768_A-12_fluid.tar.gz --no-check-certificate
tar xzvf chinese_rbt3_L-3_H-768_A-12_fluid.tar.gz
mv chinese_rbt3_L-3_H-768_A-12_fluid assets

python /qjx/PaddleHub/paddlehub/commands/hub.py install ../roberta_wwm_ext_chinese_L_24_H_1024_A_16_distillation
python test_embedding.py
