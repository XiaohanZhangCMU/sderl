#!/bin/sh

data_folder="data_wsm"

if [ ! -d ./data/${data_folder} ] ; then
    scp -r xiaohan.zhang@xiaohanzha-wsm:~/Codes/sderl/data ./data/${data_folder}
fi

python -m sderl.run plot /Users/xiaohan.zhang/Downloads/Graphics_Scratch/toplot
#/Users/xiaohan.zhang/Codes/sderl/data/${data_folder}/vpg
