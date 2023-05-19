for BC in d i n
do
for DF in 0.1 0.05 0.01 0.005 0.001
do
    python3 gaussian.py ${BC} ${DF}
done
done
