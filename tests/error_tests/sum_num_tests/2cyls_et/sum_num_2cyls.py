import sys, os
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import argparse

sys.path.append('../../../../Ncyl/')
from Ncyl import scattering, cyl

parser = argparse.ArgumentParser(description = 'verify N_p formula in Antoine for 2 cylinders')
parser.add_argument('radius1', type=float)
parser.add_argument('radius2', type=float)
parser.add_argument('spacing', type=float)
parser.add_argument('freq', type=float)

args = parser.parse_args()

grid_num = 200
pts = np.linspace(-3, 3, num = grid_num)
X, Y = np.meshgrid(pts, pts)

xl, xr, yl, yr = -3, 3, -3, 3

for tol in [1e-8, 1e-10, 1e-12, 1e-14]: 
    fig, (ax1, ax2) = plt.subplots(1, 2)

    scd = scattering([cyl(0, 'd', pos = np.array([0, args.radius1+(args.spacing/2)]), radius = args.radius1), cyl(1, 'd', pos = np.array([0, -(args.radius2 + (args.spacing/2))]), radius = args.radius2)], tol = tol, freq = args.freq)
    u_dirichlet = scd.make_u(X, Y)
    sc_ref = scattering([cyl(0, 'd', pos = np.array([0, args.radius1+(args.spacing/2)]), radius = args.radius2), cyl(1, 'd', pos = np.array([0, -(args.radius2 + (args.spacing/2))]), radius = args.radius2)], freq = args.freq, M_sum = 2*scd.M_sum)
    u_ref = sc_ref.make_u(X, Y)

    im1 = ax1.imshow(np.abs(u_dirichlet), extent=[xl, xr, xl, xr], cmap='viridis')
    ax1.set_title('abs(u), Dirichlet, M_sum = ' + str(scd.M_sum))
    fig.colorbar(im1, ax=ax1,  fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(np.log10(np.abs(u_dirichlet - u_ref)), extent=[xl, xr, xl, xr], cmap='viridis')
    ax2.set_title('dirichlet log(abs err)')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle('radius1 = ' + str(args.radius1) + ', radius2 = ' + str(args.radius2) + ', spacing = ' + str(args.spacing) + ', k = ' + str(f"{scd.k: .5e}") + ', tol = ' + str(tol) + ', cond = ' + str(f"{np.linalg.cond(scd.make_scattering_blocks()): .5e}"))

    plt.tight_layout()
    fig.set_size_inches(10, 5)
    #plt.show()
    dirname = './radius1=' + str(args.radius1) + '_radius2=' + str(args.radius2) + '_spacing=' + str(args.spacing) + '_k=' + str(f"{scd.k: .5e}") + '/'
    try:
        os.makedirs(dirname)
    except OSError:
        pass    
    
    plt.savefig(dirname + '2cyl_err_' +'radius1=' + str(args.radius1) + '_radius2=' + str(args.radius2) + '_spacing=' + str(args.spacing) + '_k=' + str(f"{scd.k: .5e}") + '_tol=' + str(tol) + '.png', dpi = 200)
    print('radius1 = ' + str(args.radius1) + ', radius2 = ' + str(args.radius2) + ', spacing = ' + str(args.spacing) + ', k = ' + str(f"{scd.k: .5e}") + ', tol = ' + str(tol) + ', cond = ' + str(f"{np.linalg.cond(scd.make_scattering_blocks()): .5e}"))
