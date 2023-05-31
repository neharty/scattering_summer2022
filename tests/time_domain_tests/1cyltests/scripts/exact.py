import numpy as np
from waveforms.waveforms import gaussian

#incident is gaussian


#exact solution for the 1 circle dirichlet case


for xpt in range(len(x)):
    for ypt in range(len(y)):
        arr = [[td.times[i], U[ypt, xpt, i]] for i in range(len(td.times))]
        np.savetxt(
            outpath+'/'+str(x[xpt])+'_'+'{:.3f}'.format(y[ypt])+'_field_exact.dat', 
            arr
            )


#exact solution for the 1 circle neumann cases
