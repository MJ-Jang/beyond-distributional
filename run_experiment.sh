EXP_TYPE=$1
TOPK=$2
BATCH_SIZE=$3


if [ $EXP_TYPE = "1" ]
then
    MODEL_TYPES="bert-base bert-large electra-small electra-large roberta-base
    roberta-large albert-base albert-large baseline"

    for mt in $MODEL_TYPES
    do
        cmd="python experiment_1.py  \
        --model_type $mt \
        --top_k=$TOPK \
        --batch_size $BATCH_SIZE"

        echo $cmd
        eval $cmd
     done
elif [ $EXP_TYPE = "2" ]
then
    MODEL_TYPES="bert-base bert-large electra-small electra-large roberta-base
    roberta-large albert-base albert-large"

        for mt in $MODEL_TYPES
    do
        cmd="python experiment_2.py  \
        --model_type $mt \
        --top_k=$TOPK \
        --batch_size $BATCH_SIZE"

        echo $cmdki9i
        eval $cmd
    done
else:
    echo "Not Implemented"
fi
