################### Running the code ####################
To run the code simply run the .sh named "run.sh". 
This will automatically send all of the other .sh files in this
directory to different nodes and the save the subsequent results here. 

################## Changing parameters that are passed ##################
In each of the .sh files (with the exception of "run.sh") you will
see a range of parameters that are to be run with the julia files. 

For example, in "neu1.sh" 
    Line 36-70: dir1.sh are meant to run the code sequentially for convergence study in the time domain.
                tt, specifies the time discretization as a power of 2. Hence tt=9 means that this specific
                run will have 512 time discretization points. Similarly, xx denotes the spatial discretization, 
                defined as a power of 2.
                T gives the final time for the simulation
                The parameter "Inc" gives the angle of the incidence pulse. There are several options
                that keep coming up in the simulations, so I hard coding the most common ones in MIX_bdry_cq.jl (line 106-121).
                Finally torder gives the different kinds of CQ methods that we use. 


    Line 79:    the "form" variable specifies the type of geometry.
                Though in most cases it is a dummy variable in this file. The
                actually geometry is defined in engine_startl.jl, where it is easier
                to define geometries and the dependent parameters for many scatterers. 

    Line 80:    specifies the center of the parametrization. Here, the array [0.0,0.0,0.0,0.0]
                will be reshaped in the julia file to be an array [[0.0,0.0],[0.0,0.0]]. 
                Again, a dummy variable, the engine_start.jl

    Line 81:    specifies the rotation applied to the parametrization. Because the parametrization
                for the open arc is different for the case of Dirichlet and Neumann, the angle
                is further specified in line 184 to 198 in "MIX_bdry_cq.jl".
                Look in engine_start.jl

    Line 82:    specifies the radius.

    Line 83:    is simply for the naming of the output files.


    Line 84:    gives the boundary condition applied to each of the parametrizations. This array
                needs to be of the same length as the array "form" in line 79. 

    Line 85:    This gives the aperture angle in line 83. Line 89 This gives the aperture angle..

                I will sometimes reuse this variable to denote something different as opposed to how it
                was used in the slotted cylindar. For example, in the case of the many strips, OP is used
                to define the variance of the Gaussian pulse (first lines of engine_start.jl). 

The parameter z, which is always set to 1 in all of the files is the over resolution
multiplier. It was used when we first started doing the CQ, but since then not. 
It should still work by changing z=1 to z=2, however I have not tested this for a while.

In line 86 the julia file is run with all of the parameters. There are two parameters
that are hard-coded:

    julia MIX_bdry_cq.jl 2 -- -- -- -- -- -- -- -- -- -- -- 0 -- -- -- 

The 2 specifies the double layer potential. Changing this to a 1 gives the single layer. 
The code will in general NOT work the with combined layer potential. 

The 0 means that the field is calculated at some collection of points. These
points are usually given in line 255 to 269 in MIX_bdry_cq.jl. It is much easier
to specify them in julia as opposed to the .jl file. Changing the 0 to a 1 will
make the code produce data for a movie (ALWAYS in a jld data format, which is easier for the movie). 
Though, to actually produce the data for a movie it is 
better to run one of the .sh titled movie. The .sh file simply gives the parameters needed
to run the movie, the code will calculate the field and save the output as a .jld file (this is 
simply a precaution). The code will also plot all of the time traces and save them in the .png file format.
Finally the code will automatically make an .mp4 that SHOULD be playable in MacOS and Windows. I have 
coded it specifically this way so it is usable for you straight off the bat. Otherwise the default
will be viewable only with VLC player. 
