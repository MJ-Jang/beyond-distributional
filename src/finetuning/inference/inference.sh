MODEL_NAME=$1
REPEATS=$2

datasets="mnli qnli qqp sst"
datasets="cola"
for data in $datasets
do
    if [ $data == "mnli" ]
    then
        cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation_matched
        --save_dir ../result/$REPEATS"
        echo $cmd
        eval $cmd

        cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation_mismatched
        --save_dir ../result/$REPEATS"
        echo $cmd
        eval $cmd

    else
        cmd="python inference.py --dataset $data --model_type $MODEL_NAME --data_type validation
        --save_dir ../result/$REPEATS"
        echo $cmd
        eval $cmd
    fi
done
