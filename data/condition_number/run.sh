for bcs in d n
do
for Nt in {6..12}
do
#large radius
radius=1
for Nmode in {1..12}
do
for space in 2.01 2.1 2.5 3 7
do
    python3 cqm_cond_num.py $Nt $Nmode $space $radius $bcs
done
done
done 


#small radius
radius=0.05
for Nmode in {1..12}
do
for space in 0.101 0.11 0.12 0.15 0.35
do
    python3 cqm_cond_num.py $Nt $Nmode $space $radius $bcs
done
done
done


