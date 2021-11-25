EXP_TYPE=$1
TOPK=$2

MODEL_TYPES="bert-base bert-large electra-base electra-large roberta-base
roberta-large albert-base albert-large"

BATCH_SIZE=32
DATA_PATH="data"
OUTP_PATH="output"


cmd="python src/experiment/experiment_baseline.py  \
    --resource_dir $DATA_PATH \
    --save_dir $OUTP_PATH \
    --thesaursus_file data/cn_thesaursus.json \
    --partial_kg_file data/conceptnet_partial.json \
    --experiment_type $EXP_TYPE"

    echo $cmd
    eval $cmd


for mt in $MODEL_TYPES
do
    cmd="python src/experiment/experiment.py  \
    --resource_dir $DATA_PATH \
    --save_dir $OUTP_PATH \
    --experiment_type $EXP_TYPE \
    --model_type $mt \
    --top_k=$TOPK \
    --batch_size $BATCH_SIZE"

    echo $cmd
    eval $cmd
done
