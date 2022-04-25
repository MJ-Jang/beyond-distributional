models="roberta-base roberta-large albert-base albert-large"

for model in $models
do
    python train.py --backbone_model_name $model --is_balanced
    rm -rf tmp_model_binary

    python train.py --backbone_model_name $model --is_balanced --freeze_enc
    rm -rf tmp_model_binary

done
