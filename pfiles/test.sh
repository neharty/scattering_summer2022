#!/bin/bash
for i in {1..10}
do
for msp in 0.1 0.5 1 1.1 1.5 2
do
    python3 cyl_pos_saver.py $i 0.5 0.05 $msp /home/nick/scattering/pfiles/test/
done
done
