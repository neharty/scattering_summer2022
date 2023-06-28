
for BC in d n
do
for DF in 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 1e-8
do
    python3 gaussian.py ${BC} ${DF}
done
done
