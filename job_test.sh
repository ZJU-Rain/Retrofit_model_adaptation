cd /your_codet5_path/sh

model=codet5_base

touch log.txt

#lang=C
#python3 run_exp.py --model_tag $model --task summarize --sub_task $lang |& tee log.txt

lang=decomC
python3 run_exp.py --model_tag $model --task summarize --sub_task $lang |& tee log.txt

lang=demiStripped
python3 run_exp.py --model_tag $model --task summarize --sub_task $lang |& tee log.txt

lang=strippedDecomC
python3 run_exp.py --model_tag $model --task summarize --sub_task $lang |& tee log.txt