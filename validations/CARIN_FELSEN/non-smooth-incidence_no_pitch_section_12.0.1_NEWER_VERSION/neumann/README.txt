################### Running the code ####################

~~~~~Trace output~~~~~
To run the code simply run the .sh named "run.sh". 
In this case, this is a single job. For this experiment
we were only interested in the trace for a fine enough discretization
to see that it matched. 

~~~~~Convergence Runs ~~~~~
If a convergence study is desired for this experiment, please
sbatch convergence.sh. This will output 22 .txt files a trace
for each observation point defined in the engine_start.jl file. 


################## Changing Parameters ##################

Line 37 gives all of the paramters that can be changed from this file.
All other paramters (line 45-52) are dummy variables. 
These variables can be changed in the engine_start.jl file. 

################## Getting the geometry in a Matlab file ###########

Simply sbatch the file called geo_to_matlab.jl to get a .mat file for the geometric 
setup.
