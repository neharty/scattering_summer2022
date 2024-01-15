Nt=12
Nmode=16
for tint in bdf2
do
for bcs in d
do
for inc in 1 2
do
for srctype in planewave
do
    export tint bcs inc Nmode Nt
    python3 4x4_datagen.py ${tint} ${Nt} ${Nmode} ${bcs} ${srctype} ${inc}
done
done
done
done
