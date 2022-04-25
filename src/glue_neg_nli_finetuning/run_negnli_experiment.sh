REPEATS=$1

models="korca/meaning-match-electra-large google/electra-large-discriminator"
models="korca/meaning-match-bert-large bert-large-cased"
models="roberta-large korca/meaning-match-roberta-large"

for model in $models
do
    cmd="bash train.sh 8 snli $model"
    echo $cmd
    eval $cmd

    cd ./inference

    cmd="python inference.py --dataset snli_neg --model_type $model --data_type validation
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd
	
    cmd="python inference.py --dataset snli --model_type $model --data_type validation
    --save_dir ../result/$REPEATS"
    echo $cmd
    eval $cmd

    cmd="python inference.py --dataset mnli --model_type $model --data_type validation_mismatched
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

