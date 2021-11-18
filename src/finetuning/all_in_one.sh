REPEATS=$1

datasets="sst mnli rte mrpc qnli qqp"
models="bert-large-cased korca/meaning-match-bert-large roberta-large korca/meaning-match-roberta-large"

for model in $models
do
    for data in $datasets
    do
        cmd="bash train.sh 8 $data $model"
        echo $cmd
        eval $cmd
    done
done

cd ./inference

for model in $models
do
    cmd="bash train.sh $model $REPEATS"
    echo $cmd
    eval $cmd
done

cd ../../
rm -rf model_binary
