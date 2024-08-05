dmin=0.25
VF=0.1
MT=cqs
CFG=/home/nh287/scattering/drivers/condnum/donut_dmin${dmin}a.cfg
export MT CFG VF dmin  
sbatch --job-name=${MT}_${VF}_${dmin}_condnum /home/nh287/scattering/drivers/condnum/setup.slurm
