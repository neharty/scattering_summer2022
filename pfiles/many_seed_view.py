import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls, plot_cyls_pub
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument('Rmax', type=float)
parser.add_argument('Rin', type=float)
parser.add_argument('rmin', type=float)
parser.add_argument('vfill', type=float)
parser.add_argument('dmin', type=float)
parser.add_argument('endseed', type=int)
args = parser.parse_args()

Rmax = args.Rmax
radius = args.rmin
vfill = args.vfill
minspace = radius*args.dmin

mainfig, mainax = plt.subplots()

for i in range(args.endseed):
    np.random.seed(i+1)
    cyls = example_setups.random_circ_hex_vol_frac(
                Rmax,
                ['d'], 
                radius, 
                vfill,
                minspace=minspace,
                bc_mode='uniform',
                rin=args.Rin
            )
    fig, ax = plot_cyls_pub(cyls)
    #get circle data
    for l in range(len(cyls)):
        x = ax.get_lines()[l].get_data()
        mainax.plot(x[0], x[1], 'k', linewidth=0.5)
    plt.close(fig)

if args.Rin > 0:
    mainax.plot(args.Rin*np.cos(np.linspace(0, 2*np.pi, 300)), args.Rin*np.sin(np.linspace(0, 2*np.pi, 300)), 'r--')

mainax.plot(Rmax*np.cos(np.linspace(0, 2*np.pi, 300)), Rmax*np.sin(np.linspace(0, 2*np.pi, 300)), 'r--')
mainax.axis('scaled')
mainax.set_xlabel('x')
mainax.set_ylabel('y')
mainfig.tight_layout()
#plt.savefig
plt.show()
