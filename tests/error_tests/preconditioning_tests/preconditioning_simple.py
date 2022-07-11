import sys, os
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import argparse

sys.path.append('../../../Ncyl/')
from Ncyl import scattering, cyl

parser = argparse.ArgumentParser(description = 'test preconditioner for 1 cyl')
parser.add_argument('radius', type=float)
parser.add_argument('freq', type=float)

args = parser.parse_args()

grid_num = 200
pts = np.linspace(-3, 3, num = grid_num)
X, Y = np.meshgrid(pts, pts)

xl, xr, yl, yr = -3, 3, -3, 3

for tol in [1e-8, 1e-10, 1e-12, 1e-14]: 
    fig, (ax1, ax2) = plt.subplots(1, 2)

    scd = scattering([cyl(0, 'd', pos = np.array([0, 0]), radius = args.radius)], tol = tol, freq = args.freq, precond = 'simple')
    u_dirichlet = scd.make_u(X, Y)
    sc_ref = scattering([cyl(0, 'd', pos = np.array([0, 0]), radius = args.radius)], tol = tol, freq = args.freq)
    u_ref = sc_ref.make_u(X, Y)
    '''
    im1 = ax1.imshow(np.abs(u_dirichlet), extent=[xl, xr, xl, xr], cmap='viridis')
    ax1.set_title('abs(u), Dirichlet, M_sum = ' + str(scd.M_sum))
    fig.colorbar(im1, ax=ax1,  fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(np.log10(np.abs(u_dirichlet - u_ref)), extent=[xl, xr, xl, xr], cmap='viridis')
    ax2.set_title('dirichlet log(abs err)')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle('radius = ' + str(args.radius) + ', k = ' + str(f"{scd.k: .5e}") + ', tol = ' + str(tol) + ', cond = ' + str(f"{np.linalg.cond(scd.scat_mat): .5e}"))

    plt.tight_layout()
    fig.set_size_inches(10, 5)
    #plt.show()
    dirname = './simple_results/'    
    try:
        os.makedirs(dirname)
    except OSError:
        pass    
    '''
    #plt.savefig(dirname + '1cyl_precond_' +'radius=' + str(args.radius) + '_k=' + str(f"{scd.k: .5e}") + '_tol=' + str(tol) + '.png', dpi = 200)
    print('radius = ' + str(args.radius) + ', k = ' + str(f"{scd.k: .5e}") + ', tol = ' + str(tol) + ', cond_precond = ' + str(f"{np.linalg.cond(scd.scat_mat): .5e}") + ', cond_old = ' + str(f"{np.linalg.cond(sc_ref.scat_mat): .5e}"))
