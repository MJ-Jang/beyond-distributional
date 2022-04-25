#!/bin/bash

NGPU=$1
DATASET=$2
MODEL_NAME=$3

cmd="python train.py  \
    task=nlp/text_classification \
    dataset=nlp/text_classification/$DATASET \
    training=glue \
    trainer=glue \
    trainer.gpus=$NGPU "


echo "Finetuning for Huggingface Model(${MODEL_NAME})"
cmd+="backbone.pretrained_model_name_or_path=$MODEL_NAME"

echo $cmd
eval $cmd

# make directory if not exist
mkdir -p ../model_binary
mkdir -p ../model_binary/korca

# copy model binaryfile to the directory we want
cp outputs/checkpoints/test.ckpt ../model_binary/$MODEL_NAME-$DATASET.ckpt
# remove outputs directory
rm -rf outputs
