for nmode in {1..15}
do
for cyld in reg rand
do
    python3 gmres_fullsolve_memory_and_timing.py ${nmode} ${cyld} /home/nick/scattering/data/proposal_imgs/s3/mem_n_time/
done
done
