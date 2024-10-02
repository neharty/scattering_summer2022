read kidx

alp=6
Np=128
Nm=80
Nt=15
dout=/home/nh287/scattering/tmatrix/data/wideband/
export alp Np Nm Nt dout

for rad in 0.5
do
for bc in d n
do
    export rad bc
    python3 /home/nh287/scattering/tmatrix/scripts/wideband/make_tmatrix_1k $alp $bc $rad $Np $Nm /home/nh287/scattering/tmatrix/kfiles/kfile_nonsmooth.dat $kidx strip /home/nh287/scattering/tmatrix/data/wideband/
done
done
