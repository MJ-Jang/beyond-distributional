REPEATS="1 2 3 4 5"
for rep in $REPEATS
do
    bash neg_nli_exp.sh $rep
done
