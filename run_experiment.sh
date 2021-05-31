EXP_TYPE=$1
TOPK=$2
BATCH_SIZE=$3

MODEL_TYPES="bert-base bert-large electra-small electra-large roberta-base
roberta-large albert-base albert-large"

for mt in $MODEL_TYPES
do
    cmd="python experiment.py  \
    --experiment_type $EXP_TYPE \
    --model_type $mt \
    --top_k=$TOPK \
    --batch_size $BATCH_SIZE"

    echo $cmd
    eval $cmd
done
