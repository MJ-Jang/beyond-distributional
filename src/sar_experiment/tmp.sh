repeats="2 3 4 5"

for rep_ in $repeats
do
    bash train.sh
   
    models="electra-small meaning_matching-albert-large-n_neg10 meaning_matching-roberta-large-n_neg10"
 #   models="meaning_matching-bert-base-n_neg10 meaning_matching-bert-large-n_neg10 meaning_matching-roberta-base-n_neg10 meaning_matching-roberta-base-n_neg10 meaning_matching-albert-base-n_neg10 meaning_matching-albert-base-n_neg10 meaning_matching-electra-small-n_neg10 meaning_matching-electra-base-n_neg10 meaning_matching-electra-large-n_neg10"
    for m in $models
    do
        bash inference.sh $m $rep_
    done
done
