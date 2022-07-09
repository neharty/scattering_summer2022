import sys
sys.path.append('../../../Ncyl')
from Ncyl import scattering, cyl
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

b = 3
radius = 0.3

sc_ref = scattering([cyl(0, 'n', pos = np.array([0, b/2]), radius = radius),cyl(1, 'd', pos = np.array([0, -b/2]), radius = radius)], M_sum = 100)
grid_num = 200
pts = np.linspace(-3, 3, num = grid_num)
X, Y = np.meshgrid(pts, pts)
u_ref = sc_ref.make_u(X, Y)

xl, xr, yl, yr = -3, 3, -3, 3

for tol in [1e-8, 1e-10, 1e-12, 1e-14]: 
    fig, ax = plt.subplots(2, 3)

    scn = scattering([cyl(0, 'n', pos = np.array([0,0]), radius = radius)], tol = tol)
    u_neumann = scn.make_u(X, Y)
    im11 = ax[0,0].imshow(np.abs(u_neumann), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[0,0].set_title('abs(u), Neumann, M_sum = ' + str(scn.M_sum))
    fig.colorbar(im11, ax=ax[0,0], fraction=0.046, pad=0.04)
    im21 = ax[1,0].imshow(np.log10(np.abs(u_neumann - scattering([cyl(0, 'n', pos = np.array([0,0]), radius = radius)], M_sum = 50).make_u(X, Y))), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[1,0].set_title('neumann log(abs err)')
    fig.colorbar(im21, ax=ax[1,0], fraction=0.046, pad=0.04)
    print("neumann bc done")

    scd = scattering([cyl(0, 'd', pos = np.array([0,0]), radius = radius)], tol = tol)
    u_dirichlet = scd.make_u(X, Y)
    im12 = ax[0,1].imshow(np.abs(u_dirichlet), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[0,1].set_title('abs(u), Dirichlet, M_sum = ' + str(scd.M_sum))
    fig.colorbar(im12, ax=ax[0,1],  fraction=0.046, pad=0.04)
    im22 = ax[1,1].imshow(np.log10(np.abs(u_dirichlet - scattering([cyl(0, 'd', pos = np.array([0,0]), radius = radius)], M_sum = 50).make_u(X, Y))), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[1,1].set_title('dirichlet log(abs err)')
    fig.colorbar(im22, ax=ax[1,1], fraction=0.046, pad=0.04)
    print("dirichlet bc done")

    sci = scattering([cyl(0, 'i', pos = np.array([0,0]), radius = radius)], tol = tol)
    u_impedance = sci.make_u(X, Y)
    im13 = ax[0,2].imshow(np.abs(u_impedance), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[0,2].set_title('abs(u), impedance, M_sum = ' + str(sci.M_sum))
    fig.colorbar(im13, ax=ax[0,2], fraction=0.046, pad=0.04)
    im23 = ax[1,2].imshow(np.log10(np.abs(u_impedance - scattering([cyl(0, 'i', pos = np.array([0,0]), radius = radius)], M_sum = 50).make_u(X, Y))), extent=[xl, xr, xl, xr], cmap='viridis')
    ax[1,2].set_title('impedance log(abs err)')
    fig.colorbar(im23, ax=ax[1,2], fraction=0.046, pad=0.04)
    print("impedance bc done")

    fig.suptitle('radius = ' + str(radius) + ', tol = ' + str(tol))

    plt.tight_layout()
    fig.set_size_inches(10, 5)
    #plt.show()
    plt.savefig('1cyl_err' + str(tol) + '.png', dpi=200)
