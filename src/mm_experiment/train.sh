MODEL_NAME=$1
N_NEG=$2

python train.py --backbone_model_name $MODEL_NAME --n_neg $N_NEG
rm -rf tmp_model_binary