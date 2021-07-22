MODEL_NAME=$1
REPEATS=$2

python inference.py --backbone_model_name $MODEL_NAME --save_dir ../../output/sei_experiment/$REPEATS --is_balanced

python inference.py --backbone_model_name $MODEL_NAME --save_dir ../../output/sei_experiment/$REPEATS --is_balanced --freeze_enc
