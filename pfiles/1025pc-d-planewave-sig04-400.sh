#!/bin/bash
for i in {1..1}
do
python3 exec_cylgen.py $i 15 g 0.01 0.1 0.4 False /home/nick/scattering/pfiles/test/
done
