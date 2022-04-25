MODEL_NAME=$1

python train.py --backbone_model_name $MODEL_NAME
rm -rf tmp_model_binary
