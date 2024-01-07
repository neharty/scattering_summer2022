import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl
from utils.plot_cyls import plot_cyls
import os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
base = Path(args.file).stem
pos = np.loadtxt(args.file)

xpos = []
ypos = []
for i in range(len(pos)):
    if i%2==0:
        xpos += [pos[i]]
    else:
        ypos += [pos[i]]
        
pos = [[xpos[i], ypos[i]] for i in range(len(pos)//2)]

cyls = [cyl('d', p, 0.5) for p in pos]
fig, ax = plot_cyls(cyls)
plt.savefig(base+'.pdf')
#plt.show()

