for radius in 0.05 0.1 1 
do
	for spacing in 0.1 1 2
	do
		for freq in 0.1 1 10 20
		do
			python3 sum_num_2cyls.py $radius $radius $spacing $freq
		done
	done
done

