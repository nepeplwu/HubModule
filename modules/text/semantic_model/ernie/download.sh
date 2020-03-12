project_path=$PWD
name=${project_path##*/}
name=${name//L_12_H_768_A_12/L-12_H-768_A-12}
name=${name//L_24_H_1024_A_16/L-24_H-1024_A-16}

hub install $name
mv ~/.paddlehub/modules/$name/assets .
mv ~/.paddlehub/modules/$name/model assets/params
rm -r ~/.paddlehub/modules/$name
hub install ../$name
