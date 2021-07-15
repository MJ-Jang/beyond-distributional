model_list="electra-large electra-small bert-base bert-large roberta-base roberta-large albert-base albert-large"


for var in $model_list
do
    cmd="python word_embedding_gen.py --model_type $var
    "
    echo $cmd
    eval $cmd
done
