REPEATS=$1

models="korca/meaning-match-bert-large korca/meaning-match-roberta-large"

for model in $models
do
    cmd="bash train.sh 8 rte $model"
    echo $cmd
    eval $cmd

    cd ./inference

    cmd="python inference.py --dataset rte_neg --model_type $model --data_type validation
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cd ../

    cmd="bash train.sh 8 mnli $model"
    echo $cmd
    eval $cmd

    cd ./inference

    cmd="python inference.py --dataset mnli_neg --model_type $model --data_type validation
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cd ../
done

