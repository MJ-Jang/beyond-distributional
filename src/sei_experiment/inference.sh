MODEL_NAME=$1

python inference.py --backbone_model_name $model --is_balanced

python inference.py --backbone_model_name $model --is_balanced --freeze_enc
