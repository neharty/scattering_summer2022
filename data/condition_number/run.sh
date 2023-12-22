bcs=d
radius=1
space=3
Nt=8
for Nmode in {1..5}
do
    python3 cqm_cond_num.py $Nt $Nmode $space $radius $bcs
done
