################### Organization of the Carin-Felsen Experiment ###################

~~~~Names of the Directories~~~~
Each of the names of the directories have been set to match the corresponding section
in the potential.pdf file.

~~~~Age of the Code~~~~
All of the directories (with the exception of the directory ending with NEWER_VERSION) are
running code that is PRE-GMRES implementation, which make the larger jobs take a little bit longer.

The newer version is running the most recent iteration (without parallelization). It is just the version of code
from the slotted cylinder with the correct incidence wave, geometry, and observation points, ect (all in engine_start.jl).

~~~~Movies~~~~
Since all of the Carin-Felsen experiments did not include movies, I have not modified the old code to be set up to produce that. 
However, there is a template folder that is capable of handling this. The only modifications needed in this folder
to run it for whatever kind of geometry and incidence wave is the engine_start.jl file. 

