for PRC in none hbs lrn antoine
do
   echo "${PRC}"
   export PRC
   sbatch runsim.slurm
done
