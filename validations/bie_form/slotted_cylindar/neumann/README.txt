################### Running the code ####################
To run the code simply run the .sh named "run.sh". 
This will automatically send all of the other .sh files in this
directory to different nodes and the save the subsequent results here. 

################## Changing parameters that are passed ##################
In each of the .sh files (with the exception of "run.sh") you will
see a range of parameters that are to be run with the julia files. 

For example, in "neu1.sh" 
    Line 83 specifies the type of geometry.
    Line 84 specifies the center of the parametrization. Here, the array [0.0,0.0,0.0,0.0]
            will be reshaped in the julia file to be an array [[0.0,0.0],[0.0,0.0]]. 
    Line 85 specifies the rotation applied to the parametrization. Because the parametrization
            for the open arc is different for the case of Dirichlet and Neumann, the angle
            is further specified in line 184 to 198 in "MIX_bdry_cq.jl".
    Line 86 specifies the radius.

    NOTE:   In most cases, the above four items will be left as dummy variables, but in this 
            case where there are only two geometries then it is simply enough to define in this
            .sh file. 

    Line 87 is simply for the naming of the output files.
    Line 89 gives the boundary condition applied to each of the paramerizations (1 for Dirichlet, 2 for Neumann). 
            This array needs to be of the same length as the array "form" in line 83. 
            This could also be specified in engine_start.jl, which is usually the case. 
    Line 89 This gives the aparture angle in line 83. Line 89 This gives the aparture angle.
            Which is further given in line 80 to 86 in the "MIX_bdry_cq.jl" file.

The parameter "Inc" gives the angle of the incidence pulse. There are several options
that keep coming up in the simulations, so I hardcoding the most common ones to be found in line 85-91 in MIX_bdry_cq.jl.   

The parameter z, which is always set to 1 in all of the files is the overresolution
multiplier. It was used when we first started doing the CQ, but since then not. 
It should still work by changing z=1 to z=2, however I have not tested this for a while.

In line 90 the julia file is run with all of the parameters. There are two parameters
that are hardcoded:

    julia MIX_bdry_cq.jl 1 -- -- -- -- -- -- -- -- -- -- -- 0 -- -- -- 

The 1 specifies the single layer potential. Changing this to a 2 gives the double layer. 
The code will NOT in general work the with combined layer potential. 

The 0 means that the field is calculated at some collection of points. These
points are usually given in "Observation Points Definition Blocks" section in engine_start.jl. 
It is much easier to specify them in julia as opposed to the .jl file. Changing the 0 to a 1 will
make the code produce data for a movie (ALWAYS in a jld data format, which is easier for the movie). 
Though, to actually produce the data for a movie it is 
better to run one of the .sh titled movie. The .sh file simply gives the parameters needed
to run the movie, the code will calculate the field and save the output as a .jld file (this is 
simply a precaution). The code will also plot all of the time traces and save them in the .png file format.
Finally the code will automatically make an .mp4 that SHOULD be playable in MacOS and windows. I have 
coded it specifically this way so it is usable for you straight off the bat. Otherwise the default
will be viewable only with VLC player. 

In addition, there are some differnt movie plotting options in line 352 in MIX_BDRY_cq.jl. 
