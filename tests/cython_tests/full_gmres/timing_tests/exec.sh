NT=12
Ncyl=10
for Nmode in {3..15}
do
    python3 nxn_timing.py $NT $Nmode $Ncyl
done
