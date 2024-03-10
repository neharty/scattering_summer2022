for TI in bdf2 irk3 irk5
do
for MEr in 1e-6 1e-10
do
for NT in {5..15}
do
        python3 conv.py ${NT} ${MEr} ${TI}
done
done
done
