
python3 dgaussian/dgaussian.py d 0
python3 dgaussian/dgaussian.py n 0

for BC in d n
do
for DF in 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8
do
    python3 gaussian/gaussian.py ${BC} ${DF}
    python3 dgaussian/dgaussian.py ${BC} ${DF}
done
done
