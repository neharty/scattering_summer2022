for NT in {10..14}
do
    export NT
    sbatch --job-name=mem_test_${NT} setup.slurm
done
