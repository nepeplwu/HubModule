wget https://paddlehub.bj.bcebos.com/model/nlp/chinese_rbtl3_L-3_H-1024_A-16_fluid.tar.gz --no-check-certificate
tar xzvf chinese_rbtl3_L-3_H-1024_A-16_fluid.tar.gz
mv chinese_rbtl3_L-3_H-1024_A-16_fluid assets

cd ..
python /qjx/PaddleHub/paddlehub/commands/hub.py install roberta_wwm_ext_chinese_L_24_H_1024_A_16_distillation
cd roberta_wwm_ext_chinese_L_24_H_1024_A_16_distillation
python test_embedding.py
