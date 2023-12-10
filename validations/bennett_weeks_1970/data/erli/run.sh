NTM=14
NMM=30
export NTM NMM
echo $NTM $NMM

for tint in bdf2 irk3 irk5
do
    #loop over nmodes at max time discretization
    for NM in {1..30}
    do
        echo $tint $NM $NTM
        python3 bw_dat_erli.py $tint $NM $NTM
    done
    
    #loop over timesteps at max Nmode
    for NT in {5..14}
    do
        echo $tint $NMM $NT
        python3 bw_dat_erli.py $tint $NMM $NT
    done
done