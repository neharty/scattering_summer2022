for dmin in 0.25 0.125
do
for VF in 0.15 0.25 0.35
do
for MT in cqs cqu rks rku
do
    CFG=/home/nh287/scattering/drivers/condnum/donut_dmin${dmin}a.cfg
    export MT CFG VF dmin  
    sbatch --job-name=${MT}_${VF}_${dmin}_condnum /home/nh287/scattering/drivers/condnum/setup.slurm
done
done
done
