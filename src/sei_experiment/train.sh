models="electra-small meaning_matching-roberta-large-n_neg10 meaning_matching-albert-large-n_neg10"
#models="meaning_matching-bert-base-n_neg10 meaning_matching-bert-large-n_neg10 meaning_matching-roberta-base-n_neg10 meaning_matching-roberta-base-n_neg10 meaning_matching-albert-base-n_neg10 meaning_matching-albert-base-n_neg10 meaning_matching-electra-small-n_neg10 meaning_matching-electra-base-n_neg10 meaning_matching-electra-large-n_neg10"
for model in $models
do
    python train.py --backbone_model_name $model --is_balanced
    rm -rf tmp_model_binary

    python train.py --backbone_model_name $model --is_balanced --freeze_enc
    rm -rf tmp_model_binary

done
