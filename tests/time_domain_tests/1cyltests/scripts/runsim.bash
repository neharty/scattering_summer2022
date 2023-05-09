for BC in d n i; do
    python3 runsim.py ${BC} 8
done

python3 plots.py

