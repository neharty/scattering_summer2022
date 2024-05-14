rm -r /home/nick/scattering/tests/freq_domain_tests/freq_dat/
python3 exec_cylgen_d_par-ann_td_and_fd.py 1 10 10 12 80 g 0.05 5 1 0.25 0.1 1.5 6 1000 False /home/nick/scattering/tests/freq_domain_tests/freq_dat/
python3 freq_domain_solver.py 1000 4 6 /home/nick/scattering/tests/freq_domain_tests/freq_dat/cyls.obj /home/nick/scattering/tests/freq_domain_tests/freq_dat/kavals.dat /home/nick/scattering/tests/freq_domain_tests/freq_dat/

