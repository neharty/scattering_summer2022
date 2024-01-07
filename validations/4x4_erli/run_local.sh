#sweep modes
Nt=14
for tint in irk5
do
for bcs in d n dndn ndnd
do
for inc in 1 2
do
for Nmode in 2 4 8 {10..16}
do
for srctype in planewave pointsrc
do
    export tint bcs inc Nmode Nt
    python3 4x4_datagen.py ${tint} ${Nt} ${Nmode} ${bcs} ${srctype} ${inc}
done
done
done
done
done 

#sweep timesteps
Nmode=16
for tint in irk5
do
for bcs in d n dndn ndnd
do
for inc in 1 2
do
for Nt in {5..14}
do
for srctype in planewave pointsrc
do
    export tint bcs inc Nmode Nt
    python3 4x4_datagen.py ${tint} ${Nt} ${Nmode} ${bcs} ${srctype} ${inc}
done
done
done
done
done
