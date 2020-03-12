project_path=$PWD
rawname=${project_path##*/}
name=${rawname//L_12_H_768_A_12/L-12_H-768_A-12}
name=${name//L_24_H_1024_A_16/L-24_H-1024_A-16}

python /qjx/PaddleHub/paddlehub/commands/hub.py install $name
mv ~/.paddlehub/modules/$rawname/assets .
mv ~/.paddlehub/modules/$rawname/model assets/params
rm -r ~/.paddlehub/modules/$rawname
python /qjx/PaddleHub/paddlehub/commands/hub.py install ../$rawname

python test_embedding.py
rm -r ~/.paddlehub/modules/$rawname
