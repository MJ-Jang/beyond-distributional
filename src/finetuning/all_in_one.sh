REPEATS=$1

datasets="sst mnli qnli qqp"
datasets="cola"
models="korca/meaning-match-electra-large google/electra-large-discriminator"
#models="korca/meaning-match-bert-large bert-large-cased"
models="google/electra-large-discriminator"
#models="korca/meaning-match-electra-large"
#models="roberta-large korca/meaning-match-roberta-large"
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
    cmd="bash inference.sh $model $REPEATS"
    echo $cmd
    eval $cmd
done

cd ../../
#rm -rf model_binary
