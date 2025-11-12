# you need to set do_train=True
cd /your_codet5_path/sh

model=codet5_base

touch log.txt

lang=decomC #programming language
python3 run_exp.py --model_tag $model --task summarize --sub_task $lang |& tee log.txt