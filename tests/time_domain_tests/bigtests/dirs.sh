for tp in gaussian deriv_of_gaussian
do
    for width in narrow match wide
    do
        for ord in ordered random
        do
            for dns in dense dilute
            do 
                for bc in impedance_small impedance_large neumann dirichlet mixed
                do
                    mkdir -p $tp/$width/$ord/$dns/$bc
                done
            done
        done
    done
done
